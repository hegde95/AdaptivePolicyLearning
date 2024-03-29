import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from SAC.utils import soft_update, hard_update
from SAC.model import GaussianPolicy, QNetwork, DeterministicPolicy, EnsembleGaussianPolicy, ConditionalQNetwork

from hyper.core import hyperActor
import numpy as np

class SAC_Agent(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hyper = args.hyper
        self.condition_q = args.condition_q
        self.steps_per_arc = args.steps_per_arc
        self.parallel = args.parallel

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device(f"cuda:{args.cuda_device}" if args.cuda else "cpu")

        if self.hyper and self.condition_q:
            self.critic = ConditionalQNetwork(num_inputs, action_space.shape[0], 4, args.hidden_size).to(device=self.device)
            self.critic_target = ConditionalQNetwork(num_inputs, action_space.shape[0], 4, args.hidden_size).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            # self.larger_policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            if self.hyper:
                self.policy =  hyperActor(action_space.shape[0], num_inputs, action_space.high[0], np.array([4,8,16,32,64,128,256,512]), meta_batch_size = args.meta_batch_size, device=self.device, search=args.search).to(self.device)
                self.policy_optim = self.policy.optimizer
            elif self.parallel:
                self.policy =  EnsembleGaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space, meta_size=args.meta_batch_size).to(self.device)
                # self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
                self.policy_optim = self.policy.current_optimizers
            else:
                if args.arc:
                    custom_arch = [int(x) for x in args.arc.split(",")]
                else:
                    custom_arch = None
                self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space, is_taper = args.taper, custom_arch=custom_arch).to(self.device)
                self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.switch_counter = 0

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def switch_policy(self, state = None):
        self.policy.change_graph(state)
        self.switch_counter = 0
        if self.parallel:
            self.policy_optim = self.policy.current_optimizers

    def select_action(self, state, evaluate=False):
        if len(state.shape) == 1:
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        else:
            state_t = torch.FloatTensor(state).to(self.device)

        if evaluate is False:
            action, _, _ = self.policy.sample(state_t)
        else:
            _, _, action = self.policy.sample(state_t)

        if len(state.shape) == 1:
            return action.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if self.condition_q and self.hyper:
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, self.policy.arcs_tensor)
            else:
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        if self.condition_q and self.hyper:
            qf1, qf2 = self.critic(state_batch, action_batch, self.policy.arcs_tensor)
        else:    
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.parallel:
            for optim in self.policy_optim:
                optim.zero_grad()
        else:
            self.policy_optim.zero_grad()

        if self.hyper or self.parallel:
            self.switch_counter += 1
            if self.switch_counter % self.steps_per_arc == 0:
                self.policy.change_graph(repeat_sample = False)
                self.switch_counter = 0
                if self.parallel:
                    self.policy_optim = self.policy.current_optimizers
            else:
                self.policy.change_graph(repeat_sample = True)

        pi, log_pi, _ = self.policy.sample(state_batch)

        if self.condition_q and self.hyper:
            qf1_pi, qf2_pi = self.critic(state_batch, pi, self.policy.arcs_tensor)
        else:
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        policy_loss.backward(retain_graph=True)
        if self.parallel:
            for optim in self.policy_optim:
                optim.step()
        else:
            self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, run_name, suffix="", ckpt_path=None, base_dir = "runs", sub_folder = "checkpoints", verbose=True):
        if not os.path.exists(f"{base_dir}/{run_name}/{sub_folder}/"):
            os.makedirs(f"{base_dir}/{run_name}/{sub_folder}/")
        if ckpt_path is None:
            ckpt_path = f"{base_dir}/{run_name}/{sub_folder}/sac_checkpoint_{suffix}"
            
        if verbose:
            print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False, add_search_params = False, device = "cpu"):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=device)

            if add_search_params:
                # Add search parameters to the checkpoint
 
                # checkpoint['policy_state_dict']['conditional_layer1_distribution.0.weight'] = self.policy.conditional_layer1_distribution[0].weight.data
                # checkpoint['policy_state_dict']['conditional_layer1_distribution.0.bias'] = self.policy.conditional_layer1_distribution[0].bias.data


                # checkpoint['policy_state_dict']['conditional_layer2_distribution.0.weight'] = self.policy.conditional_layer2_distribution[0].weight.data
                # checkpoint['policy_state_dict']['conditional_layer2_distribution.0.bias'] = self.policy.conditional_layer2_distribution[0].bias.data

                # checkpoint['policy_state_dict']['conditional_layer3_distribution.0.weight'] = self.policy.conditional_layer3_distribution[0].weight.data
                # checkpoint['policy_state_dict']['conditional_layer3_distribution.0.bias'] = self.policy.conditional_layer3_distribution[0].bias.data

                # checkpoint['policy_state_dict']['conditional_layer4_distribution.0.weight'] = self.policy.conditional_layer4_distribution[0].weight.data
                # checkpoint['policy_state_dict']['conditional_layer4_distribution.0.bias'] = self.policy.conditional_layer4_distribution[0].bias.data

                checkpoint['policy_state_dict']['conditional_arc_dist.0.weight'] = self.policy.conditional_arc_dist[0].weight.data
                checkpoint['policy_state_dict']['conditional_arc_dist.0.bias'] = self.policy.conditional_arc_dist[0].bias.data

                checkpoint['policy_state_dict']['conditional_arc_dist.2.weight'] = self.policy.conditional_arc_dist[2].weight.data
                checkpoint['policy_state_dict']['conditional_arc_dist.2.bias'] = self.policy.conditional_arc_dist[2].bias.data

            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()