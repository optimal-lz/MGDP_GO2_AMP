"""Formal single-stage Go2 perception locomotion + AMP environment."""

import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_rotate_inverse,
    torch_rand_float,
    wrap_to_pi,
)

from legged_gym.envs.random_dog.random_dog import Randomdog
from rl.MGDP.amp.state import pack_amp_state, validate_go2_dofs
from .commands import (
    amp_transition_mask,
    apply_command_deadband,
    apply_terrain_command_policy,
    classify_commands,
)
from .terrain import TERRAIN_CLASS_IDS, next_curriculum_levels


class Go2AmpRandomDog(Randomdog):
    """Narrow MGDP subclass containing only the frozen formal-task behavior."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        validate_go2_dofs(self.dof_names)

    def _init_buffers(self):
        super()._init_buffers()
        self.command_epoch = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.episode_start_position = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device)
        self.episode_start_yaw = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._base_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._termination_reward = torch.zeros_like(self._base_reward)
        self._amp_previous_state = torch.zeros(
            self.num_envs, 30, dtype=torch.float, device=self.device)

    def _get_env_origins(self):
        """Use all ten terrain types from level zero, without anchors or stages."""
        self.custom_origins = self.cfg.terrain.custom_origins
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False)
        self.env_class = torch.zeros(
            self.num_envs, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
            requires_grad=False)
        environments_per_column = self.num_envs / self.cfg.terrain.num_cols
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            environments_per_column,
            rounding_mode="floor",
        ).to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(
            device=self.device, dtype=torch.float)
        self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(
            device=self.device, dtype=torch.float)
        self.env_origins[:] = self.terrain_origins[
            self.terrain_levels, self.terrain_types]
        self.env_class[:] = self.terrain_class[
            self.terrain_levels, self.terrain_types]

    def get_amp_observations(self):
        """Return the clean, fixed-order 30D discriminator state."""
        quat = self.root_states[:, 3:7]
        lin = quat_rotate_inverse(quat, self.root_states[:, 7:10])
        ang = quat_rotate_inverse(quat, self.root_states[:, 10:13])
        return pack_amp_state(self.dof_pos, lin, ang, self.dof_vel)

    def get_idle_mask(self):
        idle, _ = classify_commands(self.commands)
        return idle

    def get_locomotion_mask(self):
        _, locomotion = classify_commands(self.commands)
        return locomotion

    def _resample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return
        count = len(env_ids)
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (count, 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (count, 1), device=self.device).squeeze(1)

        terrain_classes = self.env_class[env_ids].long()
        flat_or_rough = (
            (terrain_classes == TERRAIN_CLASS_IDS["flat"])
            | (terrain_classes == TERRAIN_CLASS_IDS["rough_flat"])
        )
        structured_ids = env_ids[~flat_or_rough]
        if structured_ids.numel() > 0:
            self.commands[structured_ids, 0] = torch_rand_float(
                self.cfg.commands.vx_deadband + 1e-6,
                self.command_ranges["lin_vel_x"][1],
                (len(structured_ids), 1), device=self.device).squeeze(1)

        sampled = apply_command_deadband(
            self.commands[env_ids], self.cfg.commands.vx_deadband,
            self.cfg.commands.yaw_deadband)
        self.commands[env_ids] = apply_terrain_command_policy(
            sampled,
            terrain_classes,
            turning_terrain_ids=(TERRAIN_CLASS_IDS["flat"],
                                 TERRAIN_CLASS_IDS["rough_flat"]),
        )
        self.command_epoch[env_ids] += 1

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done or env_ids.numel() == 0:
            return
        displacement = (self.root_states[env_ids, :2]
                        - self.episode_start_position[env_ids, :2])
        forward_direction = torch.stack((
            torch.cos(self.episode_start_yaw[env_ids]),
            torch.sin(self.episode_start_yaw[env_ids]),
        ), dim=1)
        progress = torch.sum(displacement * forward_direction, dim=1)
        forward_commands = (
            self.commands[env_ids, 0] > self.cfg.commands.vx_deadband)
        early_failure = self.reset_buf[env_ids].bool() & ~self.time_out_buf[env_ids]
        maximum_index = self.max_terrain_level - 1
        random_lower_levels = torch.randint_like(
            self.terrain_levels[env_ids], high=maximum_index)
        self.terrain_levels[env_ids] = next_curriculum_levels(
            self.terrain_levels[env_ids],
            forward_commands,
            progress,
            success_distance=self.terrain.env_length / 2,
            early_failure=early_failure,
            random_lower_levels=random_lower_levels,
            max_level=maximum_index,
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def check_termination(self):
        base_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            ) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = base_contact | self.time_out_buf

    def compute_reward(self):
        self._base_reward.zero_()
        for name, function in zip(self.reward_names, self.reward_functions):
            reward = function() * self.reward_scales[name]
            self._base_reward += reward
            self.episode_sums[name] += reward
        if self.cfg.rewards.only_positive_rewards:
            self._base_reward[:] = torch.clamp(self._base_reward, min=0.0)

        self._termination_reward.zero_()
        if "termination" in self.reward_scales:
            termination = (
                self._reward_termination() * self.reward_scales["termination"])
            self._termination_reward += termination
            self.episode_sums["termination"] += termination
        self.rew_buf[:] = self._base_reward + self._termination_reward

    def compose_total_reward(self, amp_contribution):
        """Add AMP after positive clipping and termination after AMP."""
        total = (torch.clamp(self._base_reward, min=0.0)
                 + amp_contribution + self._termination_reward)
        self.rew_buf[:] = total
        return self.rew_buf

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        reward = torch.sum(
            (torch.clamp(
                self.feet_air_time,
                max=self.cfg.rewards.amp_air_time_cap,
            ) - 0.5) * first_contact,
            dim=1,
        )
        reward *= self.commands[:, 0] > self.cfg.commands.vx_deadband
        self.feet_air_time *= ~contact_filt
        return reward

    def _reward_dof_error(self):
        error = torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        return error * self.get_idle_mask()

    def _reward_feet_edge(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        feet_pos_xy = (
            (self.rigid_body_state[:, self.feet_indices, :2]
             + self.terrain.cfg.border_size)
            / self.cfg.terrain.horizontal_scale
        ).round().long()
        feet_pos_xy[..., 0] = torch.clamp(
            feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clamp(
            feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        self.feet_at_edge = contact_filt & self.x_edge_mask[
            feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        gap_or_pit = (
            (self.env_class == TERRAIN_CLASS_IDS["gap"])
            | (self.env_class == TERRAIN_CLASS_IDS["pit"])
        )
        enabled = gap_or_pit & (self.terrain_levels > 3)
        return torch.sum(self.feet_at_edge, dim=-1).float() * enabled

    def _reward_cheat(self):
        structured = (
            (self.env_class != TERRAIN_CLASS_IDS["flat"])
            & (self.env_class != TERRAIN_CLASS_IDS["rough_flat"])
        )
        heading_error = wrap_to_pi(self.yaw - self.episode_start_yaw)
        return (structured & (torch.abs(heading_error) > 1.0)).float()

    def _reward_stuck(self):
        commanded_forward = (
            self.commands[:, 0] > self.cfg.commands.vx_deadband)
        no_forward_progress = self.base_lin_vel[:, 0] < 0.1
        return (commanded_forward & no_forward_progress).float()

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.9, 1.1, (len(env_ids), self.num_actions), device=self.device)
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        terrain_classes = self.env_class[env_ids].long()
        flat_or_rough = (
            (terrain_classes == TERRAIN_CLASS_IDS["flat"])
            | (terrain_classes == TERRAIN_CLASS_IDS["rough_flat"])
        )
        spawn_jitter = torch_rand_float(
            -0.2, 0.2, (len(env_ids), 2), device=self.device)
        spawn_jitter[~flat_or_rough, 1] = 0.0
        self.root_states[env_ids, :2] += spawn_jitter
        self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + 0.45
        self.root_states[env_ids, 7:13] = 0.0
        self.episode_start_position[env_ids] = self.root_states[env_ids, :3]
        self.episode_start_yaw[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_amp_histories(self, env_ids):
        if hasattr(self, "obs_history_buf"):
            self.obs_history_buf[env_ids] = 0.0
        if getattr(self, "image_buf", None) is not None:
            self.image_buf[env_ids] = 0.0
        if getattr(self, "image_clean_buf", None) is not None:
            self.image_clean_buf[env_ids] = 0.0

    def reset_idx(self, env_ids):
        terminal_packet = None
        terminal_diagnostics = None
        if env_ids.numel() > 0:
            terminal_packet = (
                env_ids.clone(),
                self.get_amp_observations()[env_ids].clone(),
            )
            terminal_diagnostics = {
                "env_ids": env_ids.clone(),
                "commands": self.commands[env_ids].clone(),
                "base_lin_vel": self.base_lin_vel[env_ids].clone(),
                "base_ang_vel": self.base_ang_vel[env_ids].clone(),
                "terrain_levels": self.terrain_levels[env_ids].clone(),
                "terrain_classes": self.env_class[env_ids].clone(),
                "time_outs": self.time_out_buf[env_ids].clone(),
            }
        super().reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._reset_amp_histories(env_ids)
            self._amp_previous_state[env_ids] = (
                self.get_amp_observations()[env_ids].clone())
            self.extras["terminal_amp_states"] = terminal_packet
            self.extras["terminal_diagnostics"] = terminal_diagnostics

    def step(self, actions):
        self.extras.pop("terminal_amp_states", None)
        self.extras.pop("terminal_diagnostics", None)
        locomotion_before = self.get_locomotion_mask().clone()
        epoch_before = self.command_epoch.clone()
        obs, rewards, dones, extras = super().step(actions)
        epoch_after = self.command_epoch.clone()
        extras["amp_locomotion_before"] = locomotion_before
        extras["amp_command_epoch_before"] = epoch_before
        extras["amp_command_epoch_after"] = epoch_after
        extras["amp_transition_mask"] = amp_transition_mask(
            locomotion_before, epoch_before, epoch_after, dones)
        return obs, rewards, dones, extras
