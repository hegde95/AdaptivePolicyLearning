import torch
from torch import from_numpy, device
import numpy as np
from DDPG.models import Actor, Critic
from DDPG.memory import Memory
from torch.optim import Adam
from DDPG.normalizer import Normalizer
from hyper.core import hyperActor


class Agent:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 hyper = False,
                 steps_per_arc = 20,
                 gamma=0.98):
        self.device = device("cuda")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env
        self.hyper = hyper
        self.steps_per_arc = steps_per_arc

        if self.hyper:
            self.actor = hyperActor(self.n_actions, self.n_states[0] + self.n_goals, self.action_bounds[1], np.array([4,8,16,32,64,128,256,512]), meta_batch_size = 8, device=self.device, search="False").to(self.device)
            self.actor_target = hyperActor(self.n_actions, self.n_states[0] + self.n_goals, self.action_bounds[1], np.array([4,8,16,32,64,128,256,512]), meta_batch_size = 8, device=self.device, search="False").to(self.device)
        else:
            self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
            self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)

        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.init_target_networks()
        self.tau = tau
        self.gamma = gamma

        self.capacity = capacity
        self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        if self.hyper:
            self.actor_optim = self.actor.optimizer
            self.switch_counter = 0
        else:
            self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

    def switch_policy(self, state = None):
        self.actor.change_graph(state)
        actor_graphs = [list(self.actor.list_of_arcs[i]) for i in self.actor.sampled_indices]
        self.actor_target.set_graph(actor_graphs)
        self.switch_counter = 0



    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
            goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(self.device)
            # if x.shape[0] == 1:
            #     action = self.actor(x)[0].cpu().data.numpy()
            # else:
            #     action = self.actor(x).cpu().data.numpy()
            if self.hyper:
                action = self.actor(x)[0].cpu().data.numpy()
            else:
                action = self.actor(x).cpu().data.numpy()

        if train_mode:
            action += 0.2 * np.random.randn(action.shape[0],self.n_actions)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=(action.shape[0],self.n_actions))
            action += np.random.binomial(1, 0.3, (action.shape[0],1))[0] * (random_actions - action)

        return action

    def store(self, mini_batch):
        for batch in mini_batch:
            self.memory.add(batch)
        self._update_normalizer(mini_batch)

    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self):
        states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)
        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_inputs = torch.Tensor(next_inputs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        with torch.no_grad():
            if self.hyper:
                target_q = self.critic_target(next_inputs, self.actor_target(next_inputs)[0])
            else:
                target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
            target_returns = rewards + self.gamma * target_q.detach()
            target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)

        q_eval = self.critic(inputs, actions)
        critic_loss = (target_returns - q_eval).pow(2).mean()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # self.sync_grads(self.critic)
        self.critic_optim.step()



        self.actor_optim.zero_grad()
        if self.hyper:
            self.actor.change_graph(repeat_sample = True)
            # self.switch_counter += 1
            # if self.switch_counter % self.steps_per_arc == 0:
            #     self.actor.change_graph(repeat_sample = False)
            #     self.switch_counter = 0
            # else:
            #     self.actor.change_graph(repeat_sample = True)

        a = self.actor(inputs)
        actor_loss = -self.critic(inputs, a[0] if self.hyper else a).mean()
        actor_loss += (a[0] if self.hyper else a).pow(2).mean()

        actor_loss.backward()
        # self.sync_grads(self.actor)
        self.actor_optim.step()


        return actor_loss.item(), critic_loss.item()

    def save_weights(self):
        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "state_normalizer_mean": self.state_normalizer.mean,
                    "state_normalizer_std": self.state_normalizer.std,
                    "goal_normalizer_mean": self.goal_normalizer.mean,
                    "goal_normalizer_std": self.goal_normalizer.std}, "FetchPickAndPlace.pth")

    def load_weights(self):

        checkpoint = torch.load("FetchPickAndPlace.pth")
        actor_state_dict = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state_dict)
        state_normalizer_mean = checkpoint["state_normalizer_mean"]
        self.state_normalizer.mean = state_normalizer_mean
        state_normalizer_std = checkpoint["state_normalizer_std"]
        self.state_normalizer.std = state_normalizer_std
        goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
        self.goal_normalizer.mean = goal_normalizer_mean
        goal_normalizer_std = checkpoint["goal_normalizer_std"]
        self.goal_normalizer.std = goal_normalizer_std

    def set_to_eval_mode(self):
        self.actor.eval()
        # self.critic.eval()

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)

    def _update_normalizer(self, mini_batch):
        states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()
