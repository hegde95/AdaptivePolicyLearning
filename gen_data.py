import numpy as np
import torch
from model import GaussianPolicy
import gym
import os
import pickle
import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import copy
from hyper.core import hyperActor
from hyper.ghn_modules import MLP_GHN, MlpNetwork

def get_capacity (net, inp, out):
    C = 0
    C = (inp + 2)*net[0]
    smallest_layer = net[0]
    for l in net[1:]:
        if l < smallest_layer:
            smallest_layer = l
        C += smallest_layer
    if out<smallest_layer:
        smallest_layer = out
    C+= smallest_layer
    return C



if __name__ == "__main__":
    env_name = "HalfCheetah-v2"
    ckp_file = "checkpoints/HyperAgent"

    
    env = gym.make(env_name)
    env.seed(1234)
    noisy_env = copy.deepcopy(env)
    noisy_env.seed(1234)
    num_inputs = env.observation_space.shape[0]
    env_act_dim = env.action_space.shape[0]
    action_space = env.action_space
    hidden_size = 256
    device = "cpu"

    actor = hyperActor(action_space.shape[0], num_inputs, action_space.high[0], np.arange(4,512 + 1))
    state_dict = torch.load(ckp_file, map_location='cpu')
    actor.load_state_dict(state_dict['ghn'])
    ghn = actor.ghn
    ghn = ghn.to('cpu')
    ghn.default_edges = ghn.default_edges.to('cpu')
    ghn.default_node_feat = ghn.default_node_feat.to('cpu')
    
    inp_dim = num_inputs
    out_dim = 2 * env_act_dim
    net_args = {
        'fc_layers':[],
        'inp_dim':inp_dim,
        'out_dim':out_dim,
    }  
    num_iter = 3
    num_models_to_eval = 1000

    trajs_dict = {}
    pbar = tqdm.tqdm(total=num_models_to_eval)

    for j in range(num_models_to_eval):
        num_layer = np.random.choice([1,2,3,4])
        arc = list(np.random.choice(np.arange(4,512),num_layer))
        net_args['fc_layers'] = arc 
        model = MlpNetwork(**net_args)
        shape_ind = [[0]]
        for layer in net_args['fc_layers']:
            shape_ind.append([layer])
            shape_ind.append([layer])
        shape_ind.append([out_dim])
        shape_ind.append([out_dim])    
        shape_ind = torch.Tensor(shape_ind).to('cpu')          
        ghn(model, shape_ind = shape_ind)   
        cap = get_capacity(net_args['fc_layers'], inp_dim, out_dim)       
        trajs = []
        actor_avg_rew = 0
        n_actor_avg_rew = 0
        for i in range(num_iter):
            noise = np.random.choice([10e-2, 20e-2, 30e-2, 40e-2, 50e-2])
            traj = []
            noisy_traj = []
            done = False
            state0 = env.reset()
            state1 = noisy_env.reset()
            state = np.stack((state0, state1))
            ep_reward = 0
            n_ep_reward = 0
            while not done:
                state_t = torch.FloatTensor(state)
                action_t = model(state_t)
                action_t = action_t[:,:action_t.shape[-1]//2]

                action = action_t.detach().numpy()
                og_action0 = copy.deepcopy(action[0])
                og_action1 = copy.deepcopy(action[1])
                if noise > 0:
                    action[1] += (noise * action_space.sample())

                next_state, reward, done, info = env.step(action[0])
                n_next_state, n_reward, n_done, n_info = noisy_env.step(action[1])
                ep_reward += reward
                n_ep_reward += n_reward

                traj.append((state[0], og_action0, next_state, reward, done))
                noisy_traj.append((state[1], og_action1, n_next_state, n_reward, n_done))
                state = np.stack((next_state, n_next_state))
            
            # print(f"Ep {i} reward for actor {actor_key}: {ep_reward}")
            trajs.append((traj, noisy_traj))
            actor_avg_rew += ep_reward
            n_actor_avg_rew += n_ep_reward
        trajs_dict[cap] = trajs
        # print(f"Avg reward for actor {cap}: {actor_avg_rew/num_iter}; noisy reward: {n_actor_avg_rew/num_iter} \n")
        pbar.write(f"Avg reward for actor {cap}: {actor_avg_rew/num_iter}; noisy reward: {n_actor_avg_rew/num_iter} with noise: {noise}\n")
        pbar.update(1)

    os.makedirs("trajs", exist_ok=True)
    save_path = f"trajs/ens_augmented_traj_{num_iter}"
    with open(save_path, 'wb') as f:
        pickle.dump(trajs_dict, f, pickle.HIGHEST_PROTOCOL)
    print("Done")