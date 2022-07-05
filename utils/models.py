import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6  # EPS for numerical stability

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)  # orthogonal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DoubleQNetwork(nn.Module):
    """
    Two twin Q-networks to be used as in TD3 (or DDQN). See https://arxiv.org/abs/1802.09477
    """
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(DoubleQNetwork, self).__init__()

        # Q1 architecture
        self.Q1_layers = nn.ModuleList([
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ])

        # Q2 architecture
        self.Q2_layers = nn.ModuleList([
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ])

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        out_q1 = sa.clone()
        for layer in self.Q1_layers:
            out_q1 = layer(out_q1)

        out_q2 = sa.clone()
        for layer in self.Q2_layers:
            out_q2 = layer(out_q2)

        return out_q1, out_q2


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, logstd_type, init_logstd=None, action_space=None, squash_actions=True):
        super(GaussianPolicy, self).__init__()

        self.shared_layers = nn.ModuleList([
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ])

        self.mean_linear = nn.Linear(hidden_dim, act_dim)

        # Determines whether and how the log_std should be learned
        self.logstd_type = logstd_type

        if logstd_type == "fixed":
            assert init_logstd is not None
            self.fixed_logstd_value = init_logstd
        elif logstd_type == "learned_indep_params":
            assert init_logstd is not None
            self.log_std = Parameter(init_logstd * torch.ones(size=(act_dim,), dtype=torch.float, requires_grad=True))
        elif logstd_type == "learned_state_dependent":
            assert init_logstd is None
            self.log_std_linear = nn.Linear(hidden_dim, act_dim)
        else:
            raise NotImplementedError

        # Determines whether to squash actions
        self.squash_actions = squash_actions

        # Initialise parameters
        self.apply(weights_init_)

        self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)  # small initialisation on last layer
        if logstd_type == "learned_state_dependent":
            self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)

        # for action rescaling
        if action_space is None:  # assumes action space ranging between [-1,1]
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        # Takes state as input and outputs the parameters of a diagonal, multivariate gaussian distribution
        x = state
        for layer in self.shared_layers:
            x = layer(x)

        mean = self.mean_linear(x)

        if self.logstd_type == "fixed":
            log_std = torch.tensor(self.fixed_logstd_value, dtype=torch.float).expand_as(mean).to(mean.device)
        elif self.logstd_type == "learned_indep_params":
            log_std = torch.clamp(self.log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX).expand_as(mean).to(mean.device)
        elif self.logstd_type == "learned_state_dependent":
            log_std = torch.clamp(self.log_std_linear(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            raise NotImplementedError

        return mean, log_std

    def get_dist(self, state):
        # Performs the forward pass and returns the distribution object
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        return normal, mean

    def sample(self, state):
        """
        TO SAMPLE FROM TRUNCATED GAUSSIAN POLICY:

        1. Map input x to policy mean and log-std:
             mean, log_std = neural_net(x)

        2. Reparameterization trick (to allow backprop through sample):
             u = mean + epsilon * exp(log_std),   epsilon ~ N(0,1)

        3. Truncate action domain to have legit action:
             a = tanh(u) * action_scale + action_bias

        TO COMPUTE THE PROBABILITY OF SAMPLE (a)
        * It is NOT N(a; mean, exp(log_std)) because we truncated the distribution's domain
          We need to account for that (correct the likelihood)

        1. Change of variable formula to account for truncation, shift and scale:
             pi(a|s) = mu(u|s) |det(da/du)|^-1

        2. In our particular case, this formula has a simple form.
            > The Jacobian da/du = diag( action_scale * (1 - tanh^2(u)) ) is diagonal.
            > Determinant of a diagonal matrix is the product of the elements of the diagonal.
            > Log of a product is the sum of the log() and log of a ratio is a subtraction.
           Hence:

        log pi(a|s) = log mu(u|s) - sum_{i=1...D} log( action_scale * (1 - tanh^2(u_i)) )

        See https://arxiv.org/abs/1812.05905 (Appendix C)
        """
        # Sampling from gaussian policy using the reparameterisation trick (mean + std * N(0,1))
        normal_u, mean_u = self.get_dist(state)
        u = normal_u.rsample()

        if self.squash_actions:
            # Squashing the mean and sampled action
            a = torch.tanh(u) * self.action_scale + self.action_bias
            mean_a = torch.tanh(mean_u) * self.action_scale + self.action_bias

        else:
            a = u
            mean_a = mean_u

        # Computing log probability of (squashed) action sample
        logprob_a, logprob_u = self.get_squashed_action_log_likelihood(u, normal_u)

        # We return everything because if one wants to re-evaluate the likelihood of an action a
        # with a different policy, the likelihood of the originally sampled action u will be needed.
        return a, logprob_a, mean_a, u, logprob_u, mean_u

    def get_squashed_action_log_likelihood(self, u, normal_u):
        """
        Computes the likelihood of a bounded action.
        This can be used (e.g. by PPO) to obtain the likelihood of a stored (primitive) action (u)
        given a different policy than the one that was used to sample() the action. (see self.sample())
        :param normal_u: (torch.distributions.Normal) action distribution obtained by foward propagation of state
        :param u: (torch.tensor) primitive action that was originally sampled before being squashed by tanh()
        :return: the log-likelihood of the squashed action a = tanh(u) * action_scale + action_bias
        """
        assert type(normal_u) is torch.distributions.normal.Normal
        logprob_u = normal_u.log_prob(u).sum(dim=1, keepdim=True)  # joint prob of indep gaussians is prod of gaussians

        if self.squash_actions:
            logprob_a = logprob_u - torch.log(self.action_scale * (1. - torch.tanh(u).pow(2)) + EPS).sum(dim=1, keepdim=True)
        else:
            logprob_a = logprob_u

        return logprob_a, logprob_u

    def get_entropy(self, state):
        """
        Computes the entropy of a multivariate gaussian with diagonal covariance matrix
        Used to backpropagate through the entropy (not sure whether torch.distributions allows it)
        See: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        also: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
        """
        if self.logstd_type == "learned_state_dependent":
            _, log_std = self.forward(state)
        else:
            if self.logstd_type == "fixed":
                log_std = torch.tensor(self.fixed_logstd_value, dtype=torch.float).expand_as(self.action_scale)
            elif self.logstd_type == "learned_indep_params":
                log_std = torch.clamp(self.log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            log_std = torch.unsqueeze(log_std, dim=0)

        return 0.5 * math.pow(2. * math.pi * math.e, log_std.shape[1]) * torch.log(log_std.exp().square().prod(dim=1, keepdim=True))

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
