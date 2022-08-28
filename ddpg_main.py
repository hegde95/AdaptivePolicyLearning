import gym
import numpy as np
import torch
import random
from copy import deepcopy as dc
import psutil

from DDPG.ddpg_hyper import Agent
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt





def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(2):
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
            # per_success_rate.append(info_['is_success'])
            # per_success_rate.extend([info__['is_success'] for info__ in info_])
            ep_r += r.mean()
        total_success_rate.extend(success_vector)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate)
    # global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return local_success_rate, running_r, ep_r




to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024



if __name__ == "__main__":

    ENV_NAME = "FetchPickAndPlace-v1"
    INTRO = False
    Train = True
    Play_FLAG = False
    MAX_EPOCHS = 500
    MAX_CYCLES = 50
    num_updates = 40
    MAX_EPISODES = 1
    memory_size = 7e+5 // 50
    batch_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.98
    tau = 0.05
    k_future = 4

    cuda_device_number = 3

    N = 8

    SEED = 123
    hyper = True

    env_fns = [lambda: gym.make(ENV_NAME) for _ in range(N)]
    env = SubprocVecEnv(env_fns)
    # env = gym.make(ENV_NAME)

    env.seed(SEED)
    env.action_space.seed(SEED)

    dummy_env = gym.make(ENV_NAME)

    eval_env = SubprocVecEnv(env_fns)
    eval_env.seed(SEED)
    eval_env.action_space.seed(SEED)

    state_shape = env.observation_space.spaces["observation"].shape
    n_actions = env.action_space.shape[0]
    n_goals = env.observation_space.spaces["desired_goal"].shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    agent = Agent(n_states=state_shape,
                n_actions=n_actions,
                n_goals=n_goals,
                action_bounds=action_bounds,
                capacity=memory_size,
                action_size=n_actions,
                batch_size=batch_size,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                k_future=k_future,
                env=dc(dummy_env),
                hyper=hyper,
                device_name=f"cuda:{cuda_device_number}"
                )
                
    if hyper:
        # agent.switch_policy()
        agent.actor.set_graph([[256,256,256] for _ in range(8)])
        # agent.actor_target.set_graph([[256,256,256] for _ in range(8)])

    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES):
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
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
                # if hyper:
                #     agent.switch_policy()
                # while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                #     env_dict = env.reset()
                #     state = env_dict["observation"]
                #     achieved_goal = env_dict["achieved_goal"]
                #     desired_goal = env_dict["desired_goal"]
                # while np.any([np.linalg.norm((achieved_goal - desired_goal)[l]) <= 0.05 for l in range(N)]):
                #     obs_dict = env.reset()
                #     state = env_dict["observation"]
                #     achieved_goal = env_dict["achieved_goal"]
                #     desired_goal = env_dict["desired_goal"]                    
                for t in range(50):
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, done, info = env.step(action)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]


                    # episode_dict["state"].append(state.copy())
                    # episode_dict["action"].append(action.copy())
                    # episode_dict["achieved_goal"].append(achieved_goal.copy())
                    # episode_dict["desired_goal"].append(desired_goal.copy())
                    for episode_dict, state_, action_, achieved_goal_, desired_goal_ in zip(episode_dicts, state, action, achieved_goal, desired_goal):
                        episode_dict["state"].append(state_.copy())
                        episode_dict["action"].append(action_.copy())
                        episode_dict["achieved_goal"].append(achieved_goal_.copy())
                        episode_dict["desired_goal"].append(desired_goal_.copy())


                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                # episode_dict["state"].append(state.copy())
                # episode_dict["achieved_goal"].append(achieved_goal.copy())
                # episode_dict["desired_goal"].append(desired_goal.copy())
                # episode_dict["next_state"] = episode_dict["state"][1:]
                # episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                
                for episode_dict, state_, achieved_goal_, desired_goal_ in zip(episode_dicts, state, achieved_goal, desired_goal):
                    episode_dict["state"].append(state_.copy())
                    episode_dict["achieved_goal"].append(achieved_goal_.copy())
                    episode_dict["desired_goal"].append(desired_goal_.copy())  
                    episode_dict["next_state"] = episode_dict["state"][1:]
                    episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                
                    mb.append(dc(episode_dict))

            agent.store(mb)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()
        
        # if hyper:
        #     agent.actor.scheduler.step()
        ram = psutil.virtual_memory()
        success_rate, running_reward, episode_reward = eval_agent(eval_env, agent)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        t_success_rate.append(success_rate)
        print(f"Epoch:{epoch}| "
                f"Running_reward:{running_reward[-1]:.3f}| "
                f"EP_reward:{episode_reward:.3f}| "
                # f"Memory_length:{len(agent.memory)}| "
                f"Duration:{time.time() - start_time:.3f}| "
                # f"Actor_Loss:{actor_loss:.3f}| "
                # f"Critic_Loss:{critic_loss:.3f}| "
                f"Tot_Actor_Loss:{epoch_actor_loss:.3f}| "
                f"Tot_Critic_Loss:{epoch_critic_loss:.3f}| "
                f"Actor_learning_rate:{agent.actor.scheduler.get_last_lr()[0] if hyper else actor_lr}| ",
                f"Success rate:{success_rate:.3f}| "
                f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
        # agent.save_weights()            

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate)
    plt.title("Success rate")
    plt.savefig("success_rate.png")
    plt.show()        