import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from stable_baselines3.common.vec_env import SubprocVecEnv

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=6000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--meta_batch_size', type=int, default=8, metavar='N',
                    help='hidden size (default: 8)')    
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                        help='Steps sampling random actions (default: 2000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--hyper', action="store_true",
                        help='run with a hyper network (default: False)')   
    parser.add_argument('--condition_q', action="store_true",
                        help='condition the q network with the architecture (default: False)')   
    args = parser.parse_args()
    return args

def main(args):
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    # env = gym.make(args.env_name)
    N = args.meta_batch_size
    env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
    env = SubprocVecEnv(env_fns)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.cuda = True

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    #Tesnorboard
    run_name = '{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else "",
                                                                "hyper" if args.hyper else "")
    writer = SummaryWriter('runs/' + run_name)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = np.array([False for _ in range(N)])
        state = env.reset()

        if args.hyper:
            agent.switch_policy()

        while not done.any():
            if args.start_steps > total_numsteps:
                action = np.array([env.action_space.sample() for _ in range(N)])  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += N
            total_numsteps += N
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = np.array([1 if episode_steps//N == env.get_attr('_max_episode_steps')[k] else float(not done[k]) for k in range(N)])

            memory.push_many(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', np.mean(episode_reward), i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(np.mean(episode_reward), 2)))

        if i_episode % 10 == 0 and args.eval is True:

            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = np.array([False for _ in range(N)])
                while not done.any():
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward_prechange/test', np.mean(avg_reward), i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward Pre Change: {}".format(episodes, round(np.mean(avg_reward), 2)))

            if args.hyper:
                agent.switch_policy()

                avg_reward = 0.
                episodes = 10
                for _  in range(episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = np.array([False for _ in range(N)])
                    while not done.any():
                        action = agent.select_action(state, evaluate=True)

                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward


                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes


                writer.add_scalar('avg_reward_postchange/test', np.mean(avg_reward), i_episode)

                print("Test Episodes: {}, Avg. Reward Post Change: {}".format(episodes, round(np.mean(avg_reward), 2)))
            print("----------------------------------------")


    env.close()


if __name__ == "__main__":
    args = get_args()
    main(args)    