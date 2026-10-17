import torch


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.privileged_info = None  # unknow value in the real world
            self.proprio_hist = None  # obs_history of the obs
            self.image_buf = None  # image_buf of the obs
            self.image_update_mask = None
            self.next_observations = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape,
                 Hist_info_shape, image_info_shape, device='cpu'):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape,
                                                       device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For next observations
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        # For privileged info
        self.privileged_info = torch.zeros(num_transitions_per_env, num_envs, privileged_obs_shape[0],
                                           device=self.device)

        # For hist
        self.proprio_hist = torch.zeros(num_transitions_per_env, num_envs, Hist_info_shape[0], device=self.device)

        # For image info
        self.image_buf = torch.zeros(num_transitions_per_env, num_envs, image_info_shape, device=self.device)
        self.image_update_mask = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            device=self.device, dtype=torch.bool)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)

        self.next_observations[self.step].copy_(transition.next_observations)

        # For privileged info
        self.privileged_info[self.step].copy_(transition.privileged_info)

        # For image info
        self.image_buf[self.step].copy_(transition.image_buf)
        if transition.image_update_mask is None:
            self.image_update_mask[self.step].fill_(True)
        else:
            self.image_update_mask[self.step].copy_(
                transition.image_update_mask.to(torch.bool).view(-1, 1))

        # For His
        self.proprio_hist[self.step].copy_(transition.proprio_hist)

        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None:
            return
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if hidden_states is None:
            return

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = torch.zeros(
                self.num_transitions_per_env, *hidden_states.shape,
                device=self.device, dtype=hidden_states.dtype)
        self.saved_hidden_states_a[self.step].copy_(hidden_states)

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage

            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For privileged info
        privileged_info = self.privileged_info.flatten(0, 1)
        # For image info
        image_buf = self.image_buf.flatten(0, 1)
        # For history
        proprio_hist = self.proprio_hist.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                privileged_info_batch = privileged_info[batch_idx]
                proprio_hist_batch = proprio_hist[batch_idx]
                image_buf_batch = image_buf[batch_idx]
                next_obs_batch = next_observations[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None,
                                                                                None), None, privileged_info_batch, \
                    proprio_hist_batch, image_buf_batch, next_obs_batch

    def mini_batch_generator1(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        next_observations = self.next_observations.flatten(0, 1)

        # For privileged info
        privileged_info = self.privileged_info.flatten(0, 1)
        # For image info
        image_buf = self.image_buf.flatten(0, 1)
        # For history
        proprio_hist = self.proprio_hist.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                privileged_info_batch = privileged_info[batch_idx]
                proprio_hist_batch = proprio_hist[batch_idx]
                image_buf_batch = image_buf[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None,
                                                                                None), None, privileged_info_batch, \
                    proprio_hist_batch, image_buf_batch, next_obs_batch

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Yield complete time-major sequences while shuffling environments."""
        if self.saved_hidden_states_a is None:
            raise RuntimeError("recurrent rollout is missing pre-action hidden states")
        if self.num_envs % num_mini_batches != 0:
            raise ValueError(
                "num_envs must be divisible by recurrent num_mini_batches")
        envs_per_batch = self.num_envs // num_mini_batches
        episode_starts = torch.zeros_like(self.dones, dtype=torch.bool)
        episode_starts[1:] = self.dones[:-1].to(torch.bool)

        for _ in range(num_epochs):
            environment_order = torch.randperm(
                self.num_envs, device=self.device)
            for batch_index in range(num_mini_batches):
                start = batch_index * envs_per_batch
                end = start + envs_per_batch
                env_ids = environment_order[start:end]
                masks = {
                    "image_update_mask": self.image_update_mask[:, env_ids],
                    "episode_starts": episode_starts[:, env_ids],
                }
                yield (
                    self.observations[:, env_ids],
                    self.observations[:, env_ids],
                    self.actions[:, env_ids],
                    self.values[:, env_ids],
                    self.advantages[:, env_ids],
                    self.returns[:, env_ids],
                    self.actions_log_prob[:, env_ids],
                    self.mu[:, env_ids],
                    self.sigma[:, env_ids],
                    (self.saved_hidden_states_a[0, env_ids], None),
                    masks,
                    self.privileged_info[:, env_ids],
                    self.proprio_hist[:, env_ids],
                    self.image_buf[:, env_ids],
                    self.next_observations[:, env_ids],
                )

    def reccurent_mini_batch_generator1(self, num_mini_batches, num_epochs=8):

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        indices = torch.arange(num_mini_batches * mini_batch_size, device=self.device)
        observations = self.observations.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        next_observations = self.next_observations.flatten(0, 1)

        # For privileged info
        privileged_info = self.privileged_info.flatten(0, 1)
        # For image info
        image_buf = self.image_buf.flatten(0, 1)
        # For history
        proprio_hist = self.proprio_hist.flatten(0, 1)

        # For hidden_states
        hidden_states_bacth = [hidden.permute(0, 2, 1, 3).flatten(0, 1) for hidden in self.saved_hidden_states_a]
        hid_a_batch = None

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]

                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                privileged_info_batch = privileged_info[batch_idx]
                proprio_hist_batch = proprio_hist[batch_idx]
                image_buf_batch = image_buf[batch_idx]

                hid_a_batch = [saved_hidden_states[batch_idx].transpose(1, 0).contiguous()
                               for saved_hidden_states in hidden_states_bacth]

                yield (obs_batch, None, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, None), True, \
                       privileged_info_batch, proprio_hist_batch, image_buf_batch, next_obs_batch)



