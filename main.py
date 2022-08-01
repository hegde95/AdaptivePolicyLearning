import argparse
import datetime
import logging
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
import random
import os, json


def evaluate(N, eval_env, agent):
    avg_reward = 0.
    episodes = 10
    for _  in range(episodes):
        state = eval_env.reset()
        episode_reward = 0
        done = np.array([False for _ in range(N)])
        while not done.any():
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes
    return avg_reward,episodes

def validate_run(args):
    # check if the run exists
    if not os.path.exists(os.path.join(args.base_dir, args.load_run)):
        raise ValueError("Run {} does not exist".format(args.load_run))
        
        # check if the run has a args.json file
    if not os.path.exists(os.path.join(args.base_dir, args.load_run, "args.json")):
        raise ValueError("Run {} does not have an args.json file".format(args.load_run))

        # check if the run has a tmp_stats.json file
    if not os.path.exists(os.path.join(args.base_dir, args.load_run, "tmp_stats.json")):
        raise ValueError("Run {} does not have a tmp_stats.json file".format(args.load_run))


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Adaptive Policy Learning Args')

    # general args
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--cuda_device', type=int, default=0,
                    help="sets the cuda device to run experiments on")
    parser.add_argument('--debug', action="store_true",
                        help='Will run in debug. (default: False')  

    # sac args
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
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
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 2000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    # hyper args
    parser.add_argument('--meta_batch_size', type=int, default=1, metavar='N',
                    help='hidden size (default: 1)')    
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--hyper', action="store_true",
                        help='run with a hyper network (default: False)') 
    parser.add_argument('--parallel', action="store_true",
                        help='run with an ensemble of networks (default: False)')
    parser.add_argument('--condition_q', action="store_true",
                        help='condition the q network with the architecture (default: False)')   
    parser.add_argument('--steps_per_arc', type=int, default=50, metavar='N',
                        help='steps to run between architecture samples (default: 50)')

    # logging args 
    parser.add_argument('--wandb', action="store_true",
                        help='Log to wandb. (default: False')  
    parser.add_argument('--wandb-tag', type=str, default="",
                        help='Use a custom tag for wandb. (default: "")')                        
    parser.add_argument('--save_model', action="store_true",
                    help="save the model after each episode")
    parser.add_argument('--load_run', type=str,
                        help='Load a run from latest checkpoint')
    parser.add_argument('--base_dir', type=str, default="runs",
                        help='Base directory for the experiment')

    args = parser.parse_args()
    return args

def main(args):
    # if load run is specified, load the run
    if args.load_run:
        # check if the run exists
        validate_run(args)

        # load the args
        with open(os.path.join(args.base_dir, args.load_run, "args.json"), "r") as f:
            loaded_args = json.load(f)
        # load the tmp_stats
        with open(os.path.join(args.base_dir, args.load_run, "tmp_stats.json"), "r") as f:
            loaded_tmp_stats = json.load(f)

        # create a Namespace object from the loaded args
        load_run_name = args.load_run
        args = argparse.Namespace(**loaded_args)
        args.load_run = load_run_name

        print("Found run {}".format(args.load_run))

        # load wandb id from json file
        if args.wandb:
            with open(os.path.join(args.base_dir, args.load_run, "wandb_id.json"), "r") as f:
                wandb_id = json.load(f)


        # get latest checkpoint
        checkpoint_folder = os.path.join(args.base_dir, args.load_run, "latest_model")
        list_of_ckpts = [os.path.join(checkpoint_folder,file) for file in os.listdir(checkpoint_folder)]
        # if there are no checkpoints, raise error
        if len(list_of_ckpts) == 0:
            raise ValueError("Run {} has no checkpoints".format(args.load_run))
        latest_ckpt = list_of_ckpts[-1]



    # Set the random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # INIT LOGGING

    if not args.load_run:
        run_name =  '{}_{}_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                            "DEBUG" if args.debug else "SAC", args.env_name,
                                            args.policy, "autotune" if args.automatic_entropy_tuning else "",
                                            "hyper" if args.hyper else "",
                                            str(args.seed)
                                            )                                                               
    else:
        run_name = args.load_run

    # create base directory if it doesn't exist
    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, run_name), exist_ok=True)   
                                                                 
    # tensorboard
    writer = SummaryWriter(args.base_dir + '/' + run_name)
    
    # wandb
    if args.wandb:
        tags = []
        if args.hyper:
            tags.append("hyper")
            if args.condition_q:
                tags.append("condition_q")
        elif args.parallel:
            tags.append("parallel")
        else:
            tags.append("vanilla")
        
        if args.debug:
            tags.append("debug")    

        if args.wandb_tag:
            tags.append(args.wandb_tag)   

        if not args.load_run:
            wandb_id = wandb.util.generate_id()    
            # store wandb_id as a json file
            with open(os.path.join(args.base_dir, run_name, 'wandb_id.json'), 'w') as f:
                json.dump(wandb_id, f) 
        run = wandb.init(project="AdaptivePolicyLearning", entity="khegde", name=run_name, config=args, tags = tags, id=wandb_id, resume="allow")


    # store args as a json file
    with open(os.path.join(args.base_dir, run_name, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    
    # INIT LOGIC

    # Environment
    N = args.meta_batch_size
    env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
    env = SubprocVecEnv(env_fns)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.eval:
        eval_env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
        eval_env = SubprocVecEnv(eval_env_fns)
        eval_env.seed(args.seed + 1)
        eval_env.action_space.seed(args.seed + 1)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop stats
    total_numsteps = 0
    updates = 0
    episodes_st = 0

    # if load run is specified, load the memory, total_numsteps and the agent
    if args.load_run:
        # load the memory
        memory.load_buffer(os.path.join(args.base_dir, args.load_run, "memory.pkl"))

        # load stats
        total_numsteps = loaded_tmp_stats["total_numsteps"]
        updates = loaded_tmp_stats["updates"]
        episodes_st = loaded_tmp_stats["episode"]

        # load the agent
        agent.load_checkpoint(latest_ckpt)

        print("Loaded memory, stats and model from {}".format(args.load_run))


    # MAIN LOOP
    for i_episode in itertools.count(1 + episodes_st):
        episode_reward = 0
        episode_steps = 0
        critic_1_losss, critic_2_losss, policy_losss, ent_losss, alpha_losss, updatess = [], [], [], [], [], []
        done = np.array([False for _ in range(N)])
        state = env.reset()

        # resample random policy if using hypernetwork or ensemble network
        if args.hyper or args.parallel:
            agent.switch_policy()

        # training episode loop
        while not done.any():
            if args.start_steps > total_numsteps:
                action = np.array([env.action_space.sample() for _ in range(N)])  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size: # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates) # Update parameters of all the networks
                    critic_1_losss.append(critic_1_loss)
                    critic_2_losss.append(critic_2_loss)
                    policy_losss.append(policy_loss)
                    ent_losss.append(ent_loss)
                    alpha_losss.append(alpha)
                    updatess.append(updates)                    

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


        # Log Stats
        for critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, updates in zip(critic_1_losss, critic_2_losss, policy_losss, ent_losss, alpha_losss, updatess):
            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer.add_scalar('entropy_temprature/alpha', alpha, updates)            
            if args.wandb:
                run.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss, "ent_loss": ent_loss, "alpha": alpha}, step=updates)

        writer.add_scalar('reward/train', np.mean(episode_reward), i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(np.mean(episode_reward), 2)))
        if args.wandb:
            run.log({"Train Reward": np.mean(episode_reward), "Episode": i_episode, "steps": total_numsteps}, step=total_numsteps)


        # save currents stats to a tmp file
        with open(os.path.join(args.base_dir, run_name, 'tmp_stats.json'), 'w') as f:
            json.dump({"updates": updates, "episode": i_episode, "total_numsteps": total_numsteps}, f)
        
        # save latest model
        agent.save_checkpoint(run_name = run_name, suffix="latest", base_dir=args.base_dir, sub_folder="latest_model", verbose=False)

        # save memory
        memory.save_buffer(save_path = os.path.join(args.base_dir, run_name, "memory.pkl"), env_name = args.env_name, verbose=False)

        if total_numsteps > args.num_steps:
            break

        # evaluate and save model
        if i_episode % 10 == 0 and args.eval is True:

            if args.hyper or args.parallel:
                agent.switch_policy()

            avg_reward, episodes = evaluate(N, eval_env, agent)

            writer.add_scalar('avg_reward_postchange/test', np.mean(avg_reward), i_episode)
            if args.wandb:
                run.log({"Test Reward Post Change": np.mean(avg_reward), "Episode": i_episode, "steps": total_numsteps}, step=total_numsteps)                
            print("Test Episodes: {}, Avg. Reward Post Change: {}".format(episodes, round(np.mean(avg_reward), 2)))
            print("----------------------------------------")

            if args.save_model:
                agent.save_checkpoint(run_name = run_name, suffix=total_numsteps, base_dir=args.base_dir)

    env.close()
    if args.eval:
        eval_env.close()


if __name__ == "__main__":
    args = get_args()
    main(args)    