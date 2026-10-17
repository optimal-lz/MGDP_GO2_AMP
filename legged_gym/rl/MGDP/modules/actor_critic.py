from torch.distributions import Normal
import torch.nn as nn
import torch
from .Network import *
from termcolor import cprint
import numpy as np

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        proprio_dim = kwargs['MLPModule_info']['hidden_dims'][2]
        camera_dim = kwargs.get('CNNModule_info', {}).get('output_channels', 64)
        privileged_dim = 194
        self.use_visual_memory = bool(kwargs.get('use_visual_memory', False))
        self.is_recurrent = self.use_visual_memory
        self.visual_hidden_state = None
        self.visual_gru = (
            nn.GRUCell(camera_dim, camera_dim)
            if self.use_visual_memory else None)

        self.num_actor_input = num_obs + camera_dim + proprio_dim
        self.num_critic_input = num_obs + privileged_dim

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.mlp_input = kwargs["MLPModule_info"]["input_dims"]
        self.hidden_dims = kwargs["MLPModule_info"]["hidden_dims"]
        self.mlp_encoder = MLPModule(self.mlp_input,
                                     self.hidden_dims)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        if self.visual_hidden_state is None:
            return
        if dones is None:
            self.visual_hidden_state.zero_()
            return
        done_mask = dones.to(
            device=self.visual_hidden_state.device, dtype=torch.bool).view(-1)
        self.visual_hidden_state[done_mask] = 0.0

    def ensure_hidden_state(self, visual_token):
        """Allocate rollout memory before its pre-action state is recorded."""
        if not self.use_visual_memory:
            return None
        batch_size = visual_token.shape[-2]
        if (self.visual_hidden_state is None
                or self.visual_hidden_state.shape[0] != batch_size
                or self.visual_hidden_state.device != visual_token.device):
            self.visual_hidden_state = visual_token.new_zeros(
                batch_size, visual_token.shape[-1])
        return self.visual_hidden_state

    def get_hidden_states(self):
        return self.visual_hidden_state

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs_dict, **kwargs):
        # self.update_distribution(observations)
        mean, std, _, e = self._actor_critic(obs_dict, **kwargs)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        value = self.critic(self.build_critic_input(obs_dict))
        proprioceptive_token = self.mlp_encoder(obs_dict['proprio_hist'])
        visual_token = obs_dict['image_buf'].detach()
        return value, [
            proprioceptive_token,
            visual_token[..., 0:32],
            visual_token[..., 32:],
        ]

    def _visual_memory(self, visual_token, image_update_mask,
                       hidden_states=None, episode_starts=None):
        """Apply the PPO-owned GRU only on genuine depth refreshes."""
        if not self.use_visual_memory:
            return visual_token

        is_sequence = visual_token.dim() == 3
        if not is_sequence:
            self.ensure_hidden_state(visual_token)
            hidden = (self.visual_hidden_state if hidden_states is None
                      else hidden_states)
            if episode_starts is not None:
                start = episode_starts.to(torch.bool).view(-1, 1)
                hidden = torch.where(start, torch.zeros_like(hidden), hidden)
            candidate = self.visual_gru(visual_token, hidden)
            refresh = image_update_mask.to(torch.bool).view(-1, 1)
            hidden = torch.where(refresh, candidate, hidden)
            if hidden_states is None:
                self.visual_hidden_state = hidden.detach()
            return hidden

        time_steps, batch_size, hidden_size = visual_token.shape
        hidden = (visual_token.new_zeros(batch_size, hidden_size)
                  if hidden_states is None else hidden_states)
        outputs = []
        for step in range(time_steps):
            if episode_starts is not None:
                start = episode_starts[step].to(torch.bool).view(-1, 1)
                hidden = torch.where(start, torch.zeros_like(hidden), hidden)
            candidate = self.visual_gru(visual_token[step], hidden)
            refresh = image_update_mask[step].to(torch.bool).view(-1, 1)
            hidden = torch.where(refresh, candidate, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=0)

    def build_actor_input(self, obs_dict, return_tokens=False,
                          hidden_states=None, masks=None):
        """Build deployable Actor input and stop gradients at the WM boundary."""
        proprioceptive_token = self.mlp_encoder(obs_dict['proprio_hist'])
        visual_token = obs_dict['image_buf'].detach()
        image_update_mask = obs_dict.get('image_update_mask')
        if image_update_mask is None:
            image_update_mask = torch.ones(
                *visual_token.shape[:-1], 1, device=visual_token.device,
                dtype=torch.bool)
        episode_starts = obs_dict.get('episode_starts')
        if masks is not None:
            image_update_mask = masks.get(
                'image_update_mask', image_update_mask)
            episode_starts = masks.get('episode_starts', episode_starts)
        visual_token = self._visual_memory(
            visual_token, image_update_mask, hidden_states, episode_starts)
        actor_obs = torch.cat(
            [obs_dict['obs'], proprioceptive_token, visual_token], dim=-1)
        if return_tokens:
            visual_tokens_height = visual_token[..., 0:32]
            visual_tokens_image = visual_token[..., 32:]
            return actor_obs, [
                proprioceptive_token,
                visual_tokens_height,
                visual_tokens_image,
            ]
        return actor_obs

    def build_critic_input(self, obs_dict):
        """Build the 239D asymmetric Critic input without a WM token."""
        return torch.cat(
            [obs_dict['obs'], obs_dict['privileged_info']], dim=-1)


    def _actor_critic(self, obs_dict, hidden_states=None, masks=None):
        actor_obs, extrin = self.build_actor_input(
            obs_dict, return_tokens=True, hidden_states=hidden_states,
            masks=masks)
        critic_obs = self.build_critic_input(obs_dict)

        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std

        return mu, mu * 0 + sigma, value, extrin

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


