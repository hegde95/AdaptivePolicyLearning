from main import get_args
import argparse
from sac import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
from torch import device
import numpy as np
import torch
import random
from collections import deque 
import pandas as pd
from replay_memory import ReplayMemory
import itertools




def main(args):
    args.search = True
    args.hyper = True
    device_used = device(f"cuda:{args.cuda_device}" if args.cuda else "cpu")

    N = args.meta_batch_size
    env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
    env = SubprocVecEnv(env_fns)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)    
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    size_tracker = deque(maxlen=100)
    capacity_tracker = deque(maxlen=100)
    q_tracker = deque(maxlen=100)

    df = pd.DataFrame(columns=["step", "loss", "size", "capacity"])


    # Load ckpt
    agent.load_checkpoint(args.path_to_ckpt, add_search_params = True, device = device_used)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        with torch.no_grad():
            agent.policy.change_graph(biased_sample = True)
            model_sizes = agent.policy.current_number_of_params.mean()
        done = np.array([False for _ in range(N)])
        state = env.reset()    
        while not np.any(done):
            state_t = torch.FloatTensor(state).to(device_used)
            action_t, _, _ = agent.policy.sample(state_t)
            action = action_t.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            memory.push_many(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

        print(f"\n Episode {i_episode}, Reward: {np.mean(episode_reward)}, model_size: {np.mean(model_sizes)}")

        for j in range(1000):
            state_batch, _, _, _, _ = memory.sample(batch_size=args.batch_size)

            state_batch = torch.FloatTensor(state_batch).to(device_used)
            # pc_optim.zero_grad()
            agent.policy.change_graph(biased_sample = True)
            action_batch, _, _ = agent.policy.sample(state_batch)

            arc_q1, arc_q2 = agent.critic(state_batch, action_batch)
            arc_q_loss = -torch.min(arc_q1, arc_q2).mean()



def get_apl_args(parser):

    parser.add_argument("--path_to_ckpt", type=str, 
                        help="Path to the ckpt file")

    return parser





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Adaptive Policy Learning Args')
    parser = get_args(parser)
    parser = get_apl_args(parser)
    args = parser.parse_args()
    main(args)