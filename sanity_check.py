from configs.config_helper import get_sac_args, override_config
import argparse
from SAC.sac import SAC_Agent
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
from torch import device
import numpy as np
import torch
import random
from collections import deque 
import pandas as pd
from SAC.replay_memory import ReplayMemory
import itertools
import matplotlib.pyplot as plt
import tqdm



def main(args):
    args.search = True
    args.hyper = True
    args.batch_size = 8
    device_used = device(f"cuda:{args.cuda_device}" if args.cuda else "cpu")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)  

    N = args.meta_batch_size
    env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
    env = SubprocVecEnv(env_fns)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
    eval_env = SubprocVecEnv(eval_env_fns)
    eval_env.seed(args.seed)
    eval_env.action_space.seed(args.seed)


    agent = SAC_Agent(env.observation_space.shape[0], env.action_space, args)

    size_tracker = deque(maxlen=100)
    # capacity_tracker = deque(maxlen=100)
    q_tracker = deque(maxlen=100)

    df = pd.DataFrame(columns=["step", "loss", "size"])
    updates = 0
    updates_vec = []
    eval_reward_vec = []
    eval_size_vec = []
    eval_q_vec = []


    # Load ckpt
    agent.load_checkpoint(args.path_to_ckpt, add_search_params = True, device = device_used)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    for i_episode in range(10):
        episode_reward = 0
        episode_steps = 0
        # with torch.no_grad():
        #     agent.policy.change_graph(biased_sample = False)
            # model_sizes = agent.policy.current_number_of_params.mean()
        done = np.array([False for _ in range(N)])
        state = env.reset()    
        while not np.any(done):
            state_t = torch.FloatTensor(state).to(device_used)
            with torch.no_grad():
                agent.policy.change_graph(biased_sample = False)            
            _ , _, action_t = agent.policy.sample(state_t)
            action = action_t.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            memory.push_many(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

        print(f"\n Episode {i_episode}, Reward: {np.mean(episode_reward)}")

    state_batch, _, _, _, _ = memory.sample(batch_size=512)
    state_batch = torch.FloatTensor(state_batch).to(device_used)

    for arc in tqdm.tqdm(agent.policy.list_of_arcs):
        agent.policy.set_graph(arc, 512)
        _, _, action_batch = agent.policy.sample(state_batch)
        arc_q1, arc_q2 = agent.critic(state_batch, action_batch)
        arc_q_loss = -torch.min(arc_q1, arc_q2).mean()  
        eval_size_vec.append(np.mean(agent.policy.param_counts))
        eval_q_vec.append(arc_q_loss.item())    

    # Plot
    plt.plot(eval_size_vec, eval_q_vec)
    plt.xlabel("Size")
    plt.ylabel("Q loss")
    # save the figure
    plt.savefig('figs/q_loss_size_better_arcs.png')


def get_apl_args(parser):

    parser.add_argument("--path_to_ckpt", type=str, 
                        help="Path to the ckpt file")

    return parser





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Adaptive Policy Learning Args')
    parser = get_sac_args(parser)
    parser = get_apl_args(parser)
    args = parser.parse_args()
    args = override_config(args)
    main(args)