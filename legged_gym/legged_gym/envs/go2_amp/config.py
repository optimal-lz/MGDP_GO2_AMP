"""Frozen single-stage Go2 perception locomotion + AMP configuration."""

from legged_gym.envs.random_dog.random_dog_config_stage1 import (
    RandomCfgPPOStage1,
    RandomCfgStage1,
)
from .terrain import GO2_AMP_TERRAIN_COLUMNS


AMP_EXPERT_GROUPS = {
    "forward": ["go2_forward.txt", "go2_forward_fast.txt"],
    "turn": [
        "go2_turn_left.txt",
        "go2_turn_left_fast.txt",
        "go2_turn_left_faster.txt",
        "go2_turn_right.txt",
        "go2_turn_right_fast.txt",
        "go2_turn_right_faster.txt",
    ],
}
AMP_EXPERT_GROUP_WEIGHTS = {"forward": 0.8, "turn": 0.2}


class Go2AmpCfg(RandomCfgStage1):
    class env(RandomCfgStage1.env):
        num_envs = 4096
        num_observations = 45
        # Privileged-only suffix: 3 velocity + 4 foot height + 187 height samples.
        # Actor base observations are concatenated separately by the critic.
        num_privileged_obs = 194
        num_histroy_obs = 4

    class camera(RandomCfgStage1.camera):
        resized = [16, 16]
        update_interval = 5
        wm_sensor_steps = 1
        wm_rollout_steps = 4
        noise_gaussian = 0.03
        noise_dropout = 0.1
        near_clip = 0.0
        far_clip = 5.0
        normalize = True
        mount_translation_jitter = 0.02
        mount_pitch_jitter_deg = 2.0

    class terrain(RandomCfgStage1.terrain):
        mesh_type = "mix"
        curriculum = True
        max_init_terrain_level = 0
        num_rows = 20
        num_cols = len(GO2_AMP_TERRAIN_COLUMNS)
        explicit_terrain_columns = GO2_AMP_TERRAIN_COLUMNS

    class commands(RandomCfgStage1.commands):
        curriculum = False
        heading_command = False
        resampling_time = 10.0
        vx_deadband = 0.2
        yaw_deadband = 0.2

        class ranges(RandomCfgStage1.commands.ranges):
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.0, 0.8]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-1.0, 1.0]
            new_heading = [0.0, 0.0]

    class control(RandomCfgStage1.control):
        control_type = "P"
        stiffness = {"joint": 30.0}
        damping = {"joint": 0.8}
        action_scale = 0.5
        decimation = 4

    class asset(RandomCfgStage1.asset):
        asset_name = ["go2"]
        penalize_contacts_on = ["base", "thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base"]
        penalize_contacts_on_narrow = penalize_contacts_on
        terminate_after_contacts_on_narrow = terminate_after_contacts_on

    class privInfo(RandomCfgStage1.privInfo):
        enableMeasuredVel = True
        enableFootHeight = True
        enableMeasuredHeight = True
        enablePayload = False
        enableFriction = False
        enableStiffnessDamping = False
        enableMotorStrength = False
        enablemMotorOffsets = False
        enableCom = False
        enableLimb_mass = False
        enableForce = False
        enableFootContact = False

    class domain_rand(RandomCfgStage1.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        added_friction_range = [0.5, 1.25]
        randomize_action_latency = True
        latency_range = [0.0, 0.02]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_base_mass = False
        randomize_link_mass = False
        randomize_restitution = False
        randomize_com = False
        randomize_gains = False
        randomize_motor_strength = False
        randomize_motor_offset = False
        randomize_lag_timesteps = False

    class noise(RandomCfgStage1.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(RandomCfgStage1.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class rewards(RandomCfgStage1.rewards):
        only_positive_rewards = True
        amp_air_time_cap = 0.75

        class scales(RandomCfgStage1.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            feet_air_time = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.1
            torques = -1e-4
            dof_acc = -2.5e-7
            action_rate = -0.03
            collision = -1.0
            feet_stumble = -0.1
            dof_pos_limits = -10.0
            torque_limits = -0.01
            dof_error = -0.04
            feet_edge = -1.0
            cheat = -1.0
            stuck = -1.0
            termination = 0.0
            motion_trot = 0.0
            motion_bound = 0.0
            motion_pace = 0.0
            stand_still = 0.0


class Go2AmpTrainCfg(RandomCfgPPOStage1):
    seed = 1

    class policy(RandomCfgPPOStage1.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(RandomCfgPPOStage1.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(RandomCfgPPOStage1.runner):
        runner_class_name = "Go2AmpPolicyRunner"
        num_steps_per_env = 24
        save_interval = 1000
        experiment_name = "go2_amp"
        mode = "amp"
        amp_updates_per_iter = 20
        amp_replay_rollouts = 2
        amp_batch_size = 512
        amp_learning_rate = 1e-4
        amp_gradient_penalty_coef = 10.0
        amp_groups = AMP_EXPERT_GROUPS
        amp_group_weights = AMP_EXPERT_GROUP_WEIGHTS

    class amp:
        reward_coef = 0.01


Go2AmpCfg.encoder_ppo_ref = Go2AmpTrainCfg
