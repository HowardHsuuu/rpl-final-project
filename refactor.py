import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functorch import combine_state_for_ensemble
from common import math  # Used in _estimate_value
from common.scale import RunningScale  # Used in TDMPC init

# From layers.py
class Ensemble(nn.Module):
    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return 'Vectorized ' + self._repr

# From layers.py
class ShiftAug(nn.Module):
    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

# From layers.py
class PixelPreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.).sub_(0.5)

# From layers.py
class SimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

# From layers.py
def conv(in_shape, num_channels, act=None):
    assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(), PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)

# NEW: replace hydra config for RGB-specific settings
class RGBConfig:
    def __init__(self):
        self.log_std_min = -5.0  # From original config
        self.log_std_max = 2.0   # From original config
        self.obs = 'rgb'
        self.obs_shape = {'rgb': (3, 64, 64)}
        self.action_dim = None  # To be set during initialization
        self.simnorm_dim = 50
        self.latent_dim = 50
        self.mlp_dim = 256
        self.num_q = 5
        self.horizon = 5
        self.num_samples = 512
        self.num_elites = 64
        self.temperature = 0.5
        self.tau = 0.01
        self.discount = 0.99
        self.device = 'cuda'
        self.iterations = 3
        self.min_std = 0.05
        self.max_std = 2.0
        self.num_pi_trajs = 16
        self.num_bins = 1
        self.enc_dim = 256
        self.num_enc_layers = 2
        self.num_channels = 128
        self.task_dim = 0  # No multi-task
        self.lr = 3e-4  # Used in optimizer setup
        self.enc_lr_scale = 1.0  # Used in optimizer setup
        self.grad_clip_norm = 10  # Used in gradient clipping
        self.episode_length = 100  # Need for discount calculation
        self.batch_size = 256     # Need for buffer sampling
        self.rho = 0.5           # Need for loss calculation
        self.entropy_coef = 0.1  # Need for policy loss
        self.consistency_coef = 1.0  # Need for loss calculation
        self.reward_coef = 1.0     # Need for loss calculation
        self.value_coef = 1.0      # Need for loss calculation

# From world_model.py - removed multi-task components
class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = {'rgb': conv(cfg.obs_shape['rgb'], cfg.num_channels, act=SimNorm(cfg))}
        self._dynamics = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.latent_dim),
            SimNorm(cfg)
        )
        self._reward = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.num_bins)
        )
        self._pi = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, 2 * cfg.action_dim)
        )

        self._Qs = Ensemble([nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(inplace=True),
            nn.Linear(cfg.mlp_dim, cfg.num_bins)
        ) for _ in range(cfg.num_q)])

        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    def encode(self, obs):
        return self._encoder['rgb'](obs)

    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def reward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self._reward(x)

    def pi(self, z):
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        log_pi = math.gaussian_logprob(eps, log_std)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, return_type='min', target=False):
        x = torch.cat([z, a], dim=-1)
        Qs = (self._target_Qs if target else self._Qs)(x)

        if return_type == 'min':
            return torch.min(Qs, dim=0)[0]
        elif return_type == 'mean':
            return torch.mean(Qs, dim=0)
        return Qs

    def track_q_grad(self, mode=True):
        for p in self._Qs.parameters():
            p.requires_grad_(mode)

    def soft_update_target_Q(self):
        with torch.no_grad():
            for param, target_param in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

# MODIFIED: since we're not using hydra config
def _get_discount(episode_length, discount_denom=1000, discount_min=0.95, discount_max=0.99):
    """Returns discount factor for a given episode length."""
    frac = episode_length/discount_denom
    return min(max((frac-1)/(frac), discount_min), discount_max)

class TDMPC:
    """TD-MPC agent. Implements training + inference."""
    def __init__(self, action_dim, device='cuda'):
        # MODIFIED: Using RGBConfig instead of hydra config
        self.cfg = RGBConfig()
        self.cfg.action_dim = action_dim
        self.device = torch.device(device)
        self.model = WorldModel(self.cfg).to(self.device)

        # KEPT FROM ORIGINAL
        self.optim = torch.optim.Adam([
            {'params': [p for p in self.model._encoder.values()][0].parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(self.cfg)  # KEPT FROM ORIGINAL
        self.cfg.iterations += 2*int(self.cfg.action_dim >= 20)
        self.discount = _get_discount(self.cfg.episode_length)

    # KEPT AS ORIGINAL
    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def _estimate_value(self, z, actions):
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
            z = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.discount
        return G + discount * self.model.Q(z, self.model.pi(z)[1], return_type='mean')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False):
        if t0 or not hasattr(self, '_prev_mean'):
            self._prev_mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)

        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z)[1]
                _z = self.model.next(_z, pi_actions[t])
            pi_actions[-1] = self.model.pi(_z)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]

        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
                ).clamp(-1, 1)

            # Compute elite actions
            value = self._estimate_value(z, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)
                ).clamp_(self.cfg.min_std, self.cfg.max_std)

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        z = self.model.encode(obs)
        a = self.plan(z, t0=t0, eval_mode=eval_mode)
        return a.cpu()

    def update_pi(self, zs):
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs)
        qs = self.model.Q(zs, pis, return_type='mean')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    def update(self, buffer):
        obs, action, reward = buffer.sample()  # Remove task since we're RGB-only

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:])
            td_target = reward + self.discount * self.model.Q(next_z, self.model.pi(next_z)[1], return_type='min', target=True)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0])
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t])
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, return_type='all')
        reward_preds = self.model.reward(_zs, action)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_target[t], self.cfg).mean() * self.cfg.rho**t

        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach())

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }