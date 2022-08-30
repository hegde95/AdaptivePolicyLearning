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


    # Load ckpt
    agent.load_checkpoint(args.path_to_ckpt, add_search_params = True, device = device_used)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    for i_episode in itertools.count(1):
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
            action_t, _, _ = agent.policy.sample(state_t)
            action = action_t.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            memory.push_many(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

        print(f"\n Episode {i_episode}, Reward: {np.mean(episode_reward)}")

        # evaluate
        eval_episode_reward = 0
        eval_episode_steps = 0
        avg_eval_size = 0
        done = np.array([False for _ in range(N)])
        state = eval_env.reset()
        while not np.any(done):
            state_t = torch.FloatTensor(state).to(device_used)
            with torch.no_grad():
                agent.policy.change_graph(biased_sample = True, state = state_t, eval=True)
                avg_eval_size += np.mean(agent.policy.param_counts)
            action_t, _, _ = agent.policy.sample(state_t)
            action = action_t.detach().cpu().numpy()

            next_state, reward, done, _ = eval_env.step(action)
            eval_episode_reward += reward
            eval_episode_steps += 1
            state = next_state

        print(f"Eval Episode {i_episode}, Eval Reward: {np.mean(eval_episode_reward)}, Eval Size: {avg_eval_size/eval_episode_steps}")
        updates_vec.append(updates)
        eval_reward_vec.append(np.mean(eval_episode_reward))
        eval_size_vec.append(avg_eval_size/eval_episode_steps)

        for j in range(1000):
            state_batch, _, _, _, _ = memory.sample(batch_size=args.batch_size)

            state_batch = torch.FloatTensor(state_batch).to(device_used)
            

            agent.policy.search_optimizer.zero_grad()
            agent.policy.change_graph(biased_sample = True, state = state_batch, eval=False)
            action_batch, _, _ = agent.policy.sample(state_batch)

            arc_q1, arc_q2 = agent.critic(state_batch, action_batch)
            arc_q_loss = -torch.min(arc_q1, arc_q2).mean()

            arc_q_loss.backward()
            agent.policy.search_optimizer.step()
            updates += 1

            size_tracker.append(np.mean(agent.policy.param_counts))
            q_tracker.append(arc_q_loss.item())

            df = df.append({"step": updates, "loss": np.mean(q_tracker), "size": np.mean(size_tracker)}, ignore_index=True)


        print("Average Q: {}, Average Number of Params: {}"\
            .format(np.mean(q_tracker), np.mean(size_tracker)))

        if i_episode % 2 == 0:
            fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,10))
            df.plot(x = "step", y = "size", ax = axes[0,0], c="g")
            df.plot(x = "step", y = "loss", ax = axes[0,1], c="b")
            # df.plot(x = "step", y = "capacity", ax = axes[2], c="r")
            axes[1,0].plot(updates_vec, eval_reward_vec, c = "r")
            axes[1,0].legend(["Average Eval Reward"],loc="upper right")
            axes[1,1].plot(updates_vec, eval_size_vec, c = "r")
            axes[1,1].legend(["Average Eval Size"],loc="upper right")
            fig.savefig("figs/new_better_arc_sampling.png")    



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