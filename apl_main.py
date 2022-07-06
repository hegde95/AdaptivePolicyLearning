import argparse

from torch import device
from sac import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
from replay_memory import ReplayMemory
import itertools
import numpy as np
import torch
from pcgrad import PCGrad

def get_args():
    parser = argparse.ArgumentParser(description='APL')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
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
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')

    parser.add_argument("--path_to_ckpt", type=str, default="checkpoints/HyperAgent", 
                        help="Path to the ckpt file")
    parser.add_argument('--meta_batch_size', type=int, default=1, metavar='N',
                    help='hidden size (default: 1)')   
    parser.add_argument('--hyper', action="store_true",
                        help='run with a hyper network (default: False)')   
    parser.add_argument('--condition_q', action="store_true",
                        help='condition the q network with the architecture (default: False)')   
    parser.add_argument('--steps_per_arc', type=int, default=50, metavar='N',
                        help='steps to run between architecture samples (default: 50)')

    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--cuda_device', type=int, default=0,
                    help="sets the cuda device to run experiments on")
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')                
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')  
    parser.add_argument('--search', action="store_true",
                        help='search activated (default: False)')                           
    args = parser.parse_args()
    return args

def main(args):
    args.hyper = True
    args.search = True
    device_used = device(f"cuda:{args.cuda_device}" if args.cuda else "cpu")
    N = args.meta_batch_size
    env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
    env = SubprocVecEnv(env_fns)
    # env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)


    # Load ckpt
    agent.load_checkpoint(args.path_to_ckpt, add_search_params = True, device = device_used)

    pc_optim = PCGrad(agent.policy.search_optimizer)

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
        total_q = 0
        avg_capacity = 0
        avg_num_params = 0
        while not done.any():
            state_t = torch.FloatTensor(state).to(device_used)
            
            agent.policy.change_graph(biased_sample = True)
            action_t, _, _ = agent.policy.sample(state_t)
            action = action_t.detach().cpu().numpy()


            next_state, reward, done, _ = env.step(action)
            memory.push_many(state, action, reward, next_state, done)
            episode_reward += reward
            episode_steps += 1
            total_numsteps += 1

            if total_numsteps > args.batch_size:
                # Sample a batch from memory
                state_batch, _, _, _, _ = memory.sample(batch_size=args.batch_size)

                state_batch = torch.FloatTensor(state_batch).to(device_used)
                agent.policy.change_graph(biased_sample = True)
                action_batch, _, _ = agent.policy.sample(state_batch)

                arc_q1, arc_q2 = agent.critic(state_batch, action_batch)
                arc_q_loss = -torch.min(arc_q1, arc_q2).mean()
                total_q += abs(arc_q_loss).detach().cpu().numpy()
                avg_capacity += agent.policy.current_capacites.mean()
                avg_num_params += agent.policy.current_number_of_params.mean()

                size_loss = 0.1*torch.log(agent.policy.param_counts.mean())

                if arc_q_loss > -500:
                    size_loss *= 0

                # loss = size_loss + arc_q_loss
                # loss.backward(retain_graph=True)
                pc_optim.pc_backward([size_loss, arc_q_loss])
                pc_optim.step()
                updates += 1

            state = next_state
            if total_numsteps > args.num_steps:
                break

        print("Episode: {}, Reward: {}, Steps: {}, Average Q: {}, Average Capacity: {}, Average Number of Params: {}".format(i_episode, episode_reward, episode_steps, total_q/episode_steps, avg_capacity/episode_steps, avg_num_params/episode_steps))
        if total_numsteps > args.num_steps:
            break
        if total_numsteps > args.num_steps:
            break

        if i_episode % 10 == 0:
            agent.save_checkpoint(run_name = args.path_to_ckpt + "_apl", suffix=total_numsteps)

    print("Total Steps: {}".format(total_numsteps))

    
if __name__ == '__main__':
    args = get_args()
    main(args)

     