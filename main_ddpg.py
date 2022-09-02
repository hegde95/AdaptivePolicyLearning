import argparse
from copy import deepcopy as dc
import datetime
import logging
import gym
import numpy as np
import itertools
import torch
from DDPG.ddpg_hyper import Agent

from torch.utils.tensorboard import SummaryWriter
from SAC.replay_memory import ReplayMemory
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
import random
import os, json
from configs.config_helper import get_ddpg_args, override_config
import dmc2gym
import time

def evaluate(env_, agent_, N):
    total_success_rate = []
    running_r = []
    if agent_.hyper:
        agent_.actor.set_graph([[4,4,4],[8,8,8],[16,16,16],[32,32,32],[64,64,64],[128,128,128],[256,256,256],[512,512,512]])
    for ep in range(8):
        per_success_rate = []
        env_dictionary = env_.reset()
        success_vector = np.zeros(N)
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        # while np.linalg.norm(ag - g) <= 0.05:
        while np.any([np.linalg.norm((ag - g)[i])<=0.05 for i in range(N)]):
            env_dictionary = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0
        for t in range(50):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, _, info_ = env_.step(a)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            success_vector = np.array([info__['is_success'] + success_vector[ind] for ind, info__ in enumerate(info_)]).astype(bool)
            ep_r += r.mean()
        total_success_rate.extend(success_vector)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate)
    return local_success_rate, running_r, ep_r


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
        latest_model_folder = os.path.join(args.base_dir, args.load_run, "latest_model")
        list_of_ckpts = [os.path.join(latest_model_folder,file) for file in os.listdir(latest_model_folder)]

        # if there are no checkpoints, raise error
        if len(list_of_ckpts) == 0:
            raise ValueError("Run {} has no checkpoints".format(args.load_run))
        for file in list_of_ckpts:
            if "backup" in file:
                latest_ckpt_backup = file
            else:
                latest_ckpt = file



    # Set the random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # INIT LOGGING

    if not args.load_run:
        run_name =  '{}_{}_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                            "DEBUG" if args.debug else "DDPG", 
                                            "dm" + args.domain + args.task if args.dm_control else args.env_name,
                                            "", "",
                                            "hyper" if args.hyper else "",
                                            str(args.seed)
                                            )                                                               
    else:
        run_name = args.load_run

    # create base directory if it doesn't exist
    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, run_name), exist_ok=True) 
    os.makedirs(os.path.join(args.base_dir, run_name, "latest_model"), exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, run_name, "latest_memory"), exist_ok=True)  

    # tensorboard
    writer = SummaryWriter(args.base_dir + '/' + run_name)
    
    # wandb
    if args.wandb:
        tags = []
        if args.hyper:
            tags.append("hyper")
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
    if args.dm_control:
        env_fns = [lambda: dmc2gym.make(domain_name=args.domain, task_name=args.task, seed=args.seed) for _ in range(N)]
        dummy_env = dmc2gym.make(domain_name=args.domain, task_name=args.task, seed=args.seed)
    else:
        env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
        dummy_env = gym.make(args.env_name)

    env = SubprocVecEnv(env_fns)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.eval:
        if args.dm_control:
            eval_env_fns = [lambda: dmc2gym.make(domain_name=args.domain, task_name=args.task, seed=args.seed + 1) for _ in range(N)]
        else:
            eval_env_fns = [lambda: gym.make(args.env_name) for _ in range(N)]
        eval_env = SubprocVecEnv(eval_env_fns)
        eval_env.seed(args.seed + 1)
        eval_env.action_space.seed(args.seed + 1)

    state_shape = env.observation_space.spaces["observation"].shape
    n_actions = env.action_space.shape[0]
    n_goals = env.observation_space.spaces["desired_goal"].shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]


    agent = Agent(n_states=state_shape,
                n_actions=n_actions,
                n_goals=n_goals,
                action_bounds=action_bounds,
                capacity=args.memory_size,
                action_size=n_actions,
                batch_size=args.batch_size,
                actor_lr=args.lr,
                critic_lr=args.lr,
                gamma=args.gamma,
                tau=args.tau,
                k_future=args.k_future,
                env=dc(dummy_env),
                hyper=args.hyper,
                steps_per_arc = args.steps_per_arc,
                device_name=f"cuda:{args.cuda_device}"
                )

    # Training Loop stats
    total_numsteps = 0
    # updates = 0
    episodes_st = 0
    MAX_EPOCHS = 5000
    MAX_CYCLES = 50
    MAX_EPISODES = 8
    num_updates = 40
    t_success_rate, total_ac_loss, total_cr_loss = [], [], []
    epochs_run = 0


    # if load run is specified, load the memory, total_numsteps and the agent
    if args.load_run:
        # load the memory
        try:
            agent.memory.load_buffer(os.path.join(args.base_dir, args.load_run, "latest_memory", "memory.pkl"))
            print("Loaded memory from {}".format(os.path.join(args.base_dir, args.load_run, "latest_memory", "memory.pkl")))
        except:
            agent.memory.load_buffer(os.path.join(args.base_dir, args.load_run, "latest_memory", "memory_backup.pkl"))
            print("Could not load memory from {}. Loaded from {}.".format(os.path.join(args.base_dir, args.load_run, "latest_memory", "memory.pkl"), os.path.join(args.base_dir, args.load_run, "latest_memory", "memory_backup.pkl")))
        
        # load stats
        total_numsteps = loaded_tmp_stats["total_numsteps"]
        epochs_run = loaded_tmp_stats["epoch"] + 1
        print("Loaded stats from {}".format(os.path.join(args.base_dir, args.load_run, "tmp_stats.json")))

        # load the agent
        try:
            agent.load_checkpoint(latest_ckpt)
            print("Loaded agent from {}".format(latest_ckpt))
        except:
            agent.load_checkpoint(latest_ckpt_backup)
            print("Could not load agent from {}. Loaded from {}.".format(latest_ckpt, latest_ckpt_backup))

        agent.actor.scheduler.last_epoch = epochs_run
        agent.actor.scheduler._step_count = epochs_run + 1
        

    # MAIN LOOP
    # for each epoch, run multiple cycles of rollout collection and training
    # at end of each epoch eval (and) save.
    for epoch in range(epochs_run, MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss, epoch_critic_loss = 0, 0

        # for each cycle, collect MAX_EPISODES rollout data
        # at end of each cycle, train the model 
        for cycle in range(0, MAX_CYCLES):
            mb = []
            cycle_actor_loss, cycle_critic_loss = 0, 0

            for episode in range(MAX_EPISODES):
                episode_dicts = [{
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []} for _ in range(N)]
                env_dict = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]

                if args.hyper:
                    agent.switch_policy()

                # collect rollouts
                for t in range(50):
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, done, info = env.step(action)
                    total_numsteps += N

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]
                    for episode_dict, state_, action_, achieved_goal_, desired_goal_ in zip(episode_dicts, state, action, achieved_goal, desired_goal):
                        episode_dict["state"].append(state_.copy())
                        episode_dict["action"].append(action_.copy())
                        episode_dict["achieved_goal"].append(achieved_goal_.copy())
                        episode_dict["desired_goal"].append(desired_goal_.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()
                
                for episode_dict, state_, achieved_goal_, desired_goal_ in zip(episode_dicts, state, achieved_goal, desired_goal):
                    episode_dict["state"].append(state_.copy())
                    episode_dict["achieved_goal"].append(achieved_goal_.copy())
                    episode_dict["desired_goal"].append(desired_goal_.copy())  
                    episode_dict["next_state"] = episode_dict["state"][1:]
                    episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]                
                    mb.append(dc(episode_dict))

            agent.store(mb)

            # update
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()

        if args.hyper:
            # step the LR scheduler for the hyper network
            agent.actor.scheduler.step() 

        success_rate, running_reward, episode_reward = evaluate(eval_env, agent, N)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        t_success_rate.append(success_rate)

        actor_learning_rate = agent.actor.scheduler.get_last_lr()[0] if args.hyper else args.lr
        writer.add_scalar('loss/critic', epoch_critic_loss, epoch)
        writer.add_scalar('loss/actor', epoch_actor_loss, epoch)
        writer.add_scalar('success_rate', success_rate, epoch)
        writer.add_scalar('Actor_learning_rate', actor_learning_rate, epoch)

        if args.wandb:
            run.log({"critic_loss": epoch_critic_loss, "actor_loss": epoch_actor_loss, "success_rate": success_rate, "actor_learning_rate":actor_learning_rate, "epoch": epoch, "steps": total_numsteps}, step=total_numsteps)

        print(f"Epoch:{epoch}| "
                f"Running_reward:{running_reward[-1]:.3f}| "
                f"EP_reward:{episode_reward:.3f}| "
                f"Duration:{time.time() - start_time:.3f}| "
                f"Tot_Actor_Loss:{epoch_actor_loss:.3f}| "
                f"Tot_Critic_Loss:{epoch_critic_loss:.3f}| "
                f"Actor_learning_rate:{actor_learning_rate}| ",
                f"Success rate:{success_rate:.3f}| ")

        # save currents stats to a tmp file
        with open(os.path.join(args.base_dir, run_name, 'tmp_stats.json'), 'w') as f:
            json.dump({"epoch": epoch, "total_numsteps": total_numsteps}, f)

        # check if a latest model exists and rename it to backup
        latest_ckpt = os.path.join(args.base_dir, run_name, 'latest_model', 'sac_checkpoint_latest')
        if os.path.exists(latest_ckpt):
            os.rename(latest_ckpt, latest_ckpt + '_backup')
        # save latest model
        agent.save_checkpoint(run_name = run_name, suffix="latest", base_dir=args.base_dir, sub_folder="latest_model", verbose=False)

        # check if a memory file exists and rename it to backup
        memory_file = os.path.join(args.base_dir, run_name, "latest_memory", 'memory.pkl')
        if os.path.exists(memory_file):
            os.rename(memory_file, memory_file.split(".")[0] + '_backup.pkl')
        # save memory
        agent.memory.save_buffer(save_path = os.path.join(args.base_dir, run_name, "latest_memory", "memory.pkl"), env_name = args.env_name, verbose=False)

        if args.save_model and epoch % 10 == 0:
            agent.save_checkpoint(run_name = run_name, suffix=total_numsteps, base_dir=args.base_dir, verbose = False)

        if total_numsteps > args.num_steps:
            break

    env.close()
    if args.eval:
        eval_env.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Adaptive Policy Learning Args')
    parser = get_ddpg_args(parser)
    args = parser.parse_args()
    args = override_config(args, "ddpg")
    main(args)    