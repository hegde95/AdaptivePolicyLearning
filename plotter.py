import warnings
warnings.filterwarnings("ignore")
import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
import numpy as np
import torch
# from hyper.loader import MLPNets1M
from hyper.ghn_modules import MLP_GHN, MlpNetwork
from hyper.core import hyperActor
import time
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats
import d4rl


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    # parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    # parser.add_argument("--hours", type=int, default=1, help="Hours to run the plotter script for")
    parser.add_argument("--path_to_ckpt", type=str, default="checkpoints/HyperAgent", help="Path to the ckpt file")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--env", type=str, default="HalfCheetah-v2", help="Which environment to run eval over")

    args = parser.parse_args()
    return args

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


def evaluate_for_single_policy(env, model):
    reward_batch = []
    state = env.reset()
    rewards = 0
    while True:
        state = torch.from_numpy(state).float().to('cpu')
        action = model(state)
        action = action[:,:action.shape[-1]//2]
        action = action.detach().cpu().numpy()
        state, reward, done, _ = env.step(action)
        rewards += reward
        if done.any():
            break
    reward_batch.append(rewards)
    return np.mean(reward_batch)

def cal_prob(fc_layer,prob_dict):
    model_prob = 1
    for j in range(4):
        if j < len(fc_layer):
            layer_prob = prob_dict[f"layer{j+1}"][fc_layer[j]]
        else:
            layer_prob = prob_dict[f"layer{j+1}"][0]
        model_prob *= layer_prob
    return model_prob

def evaluate(env, ghn, model_dict, list_of_net_arch_eval): 
    """
    Makes an evaluation run with the current GHN
    """
    ghn.eval()
    inp_dim = env.observation_space.shape[0]
    out_dim = 2 *env.action_space.shape[0]
    net_args = {
        'fc_layers':[],
        'inp_dim':inp_dim,
        'out_dim':out_dim,
    }      
    hyper_reward = 0
    for i,net_arch in tqdm.tqdm(enumerate(list_of_net_arch_eval)):
        net_args['fc_layers'] = net_arch    
        model = MlpNetwork(**net_args)
        shape_ind = [[0]]
        for layer in net_args['fc_layers']:
            shape_ind.append([layer])
            shape_ind.append([layer])
        shape_ind.append([out_dim])
        shape_ind.append([out_dim])    
        shape_ind = torch.Tensor(shape_ind).to('cpu')          
        ghn(model, shape_ind = shape_ind)  
        reward_per_policy = evaluate_for_single_policy(env, model)

        model_dict.at[model_dict.index[model_dict['architecture'] == str(net_args['fc_layers'])][0], 'reward'] = reward_per_policy        
        hyper_reward += reward_per_policy

    return hyper_reward/len(list_of_net_arch_eval)

def get_plots_lims(env_name):
    if env_name in ["Ant-v3", "Ant-v2"]:
        return -200, 6500, 5, 12, 5, 14
    elif env_name in ["HalfCheetah-v3", "HalfCheetah-v2"]:
        return -200, 14000, 4, 10, 5, 14
    elif env_name in ["Walker2d-v3", "Walker2d-v2"]:
        return -200, 6000, 4, 10, 5, 14
    elif env_name in ["Hopper-v3", "Hopper-v2"]:
        return -200, 4000, 3, 10, 4, 14
    elif env_name in ["Swimmer-v3", "Swimmer-v2"]:
        return -30, 200, 3, 9, 4, 14
    elif env_name in ["InvertedDoublePendulum-v2", "InvertedDoublePendulum-v3"]:
        return -30, 10000, 3, 10, 3, 14
    elif env_name in ["InvertedPendulum-v2", "InvertedPendulum-v3"]:
        return -30, 1100, 3, 9, 4, 14
    else:
        raise ValueError('Given Env has no plot limits.')

def plot_model_dict(model_dict, env_name, i, rvp_folder, rvc_folder):
    ylim_l, ylim_h, c_low, c_high, p_low, p_high = get_plots_lims(env_name)
    # ylim_l = -0.2
    # ylim_h = 1.5
    model_dict = model_dict.sort_values(by = ['capacity'], ascending = True)
    # model_dict = model_dict.replace([np.inf, -np.inf], np.nan).dropna()
    model_dict =model_dict[model_dict['capacity'] != 0]
    # capacity_bins = get_capacity_bins(env_name)
    bin_means, bin_edges, binnumber = stats.binned_statistic(np.log(model_dict['capacity']), model_dict['reward'], statistic='mean', bins=8)
    bin_std, _, _ = stats.binned_statistic(np.log(model_dict['capacity']), model_dict['reward'], statistic='std', bins=8)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    # save image
    plt.scatter(np.log(model_dict['params']), model_dict['reward'], c=np.log(model_dict['capacity']), s=10, cmap = 'coolwarm')
    plt.xlabel('log(num params)')
    plt.ylabel('reward')
    plt.xlim(p_low,p_high)
    plt.ylim(ylim_l, ylim_h)
    plt.title(f"{i} Epochs")
    plt.colorbar().set_label('log(capacity)', rotation=270, labelpad = 10)
    plt.savefig(rvp_folder + '/' + str(i) + '.png', dpi = 300)
    plt.clf()

    model_dict = model_dict.sort_values(by = ['params'], ascending = True)

    plt.plot(bin_centers, bin_means)
    plt.scatter(np.log(model_dict['capacity']), model_dict['reward'], c=np.log(model_dict['params']), s=10, cmap = 'coolwarm')
    plt.xlabel('log(capacity)')
    plt.ylabel('reward')
    plt.xlim(c_low,c_high)
    plt.ylim(ylim_l, ylim_h)
    plt.title(f"{i} Epochs")
    plt.colorbar().set_label('log(num params)', rotation=270, labelpad = 10)
    # plt.plot(np.log(model_dict['capacity']), gaussian_filter1d(model_dict['norm_reward'], sigma =80))
    plt.savefig(rvc_folder + '/' + str(i) + '.png', dpi = 300)
    plt.clf()


def get_model_dict(num_models_to_eval, list_of_allowable_layers, env_obs_dim, env_act_dim, env, ghn):
    ghn = ghn.to('cpu')
    ghn.default_edges = ghn.default_edges.to('cpu')
    ghn.default_node_feat = ghn.default_node_feat.to('cpu')
    
    inp_dim = env_obs_dim
    out_dim = 2 * env_act_dim
    net_args = {
        'fc_layers':[],
        'inp_dim':inp_dim,
        'out_dim':out_dim,
    }    
    # list_of_arcs = []
    model_dict = pd.DataFrame(columns=['architecture', 'reward', 'params', 'capacity', 'probability'])
    for i in tqdm.tqdm(range(num_models_to_eval)):
        num_layer = np.random.choice([1,2,3,4])
        arc = list(np.random.choice(list_of_allowable_layers,num_layer))
        # list_of_arcs.append(arc)
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
        reward_per_policy = evaluate_for_single_policy(env, model)
        # rew = np.float64(0)
        params = sum(p.numel() for p in model.parameters())
        cap = get_capacity(net_args['fc_layers'], inp_dim, out_dim)
        # model_prob = cal_prob(net_args['fc_layers'],prob_dict)  
        # model_prob += 1e-15

        model_dict  = pd.concat([model_dict, pd.DataFrame({'architecture':[str(arc)],'reward':[reward_per_policy], 'params':[params], 'capacity':[cap]})], ignore_index=True)

    model_dict['params'] = model_dict['params'].astype(np.float64)
    model_dict['capacity'] = model_dict['capacity'].astype(np.float64)
    model_dict['reward'] = model_dict['reward'].astype(np.float64)
    # model_dict['probability'] = model_dict['probability'].astype(np.float64)
    model_dict = model_dict.sort_values(by = ['params'], ascending= False)
    model_dict = model_dict.sort_values(by = ['capacity'], ascending = False)
    return model_dict


def main(config):
    num_models_to_eval = 10000
    list_of_allowable_layers = np.arange(4,512)

    # if "halfcheetah" in config.run_name:
    #     actual_env_id = "HalfCheetah-v2"
    # elif "hopper" in config.run_name:
    #     actual_env_id = "Hopper-v2"
    # elif "walker" in config.run_name:
    #     actual_env_id = "Walker2d-v2"
    # elif "ant" in config.run_name:
    #     actual_env_id = "Ant-v2"    

    actual_env_id = config.env

    eval_env_fns = [lambda: gym.make(actual_env_id) for _ in range(5)]
    eval_envs = SubprocVecEnv(eval_env_fns)
    eval_envs.seed(config.seed)

    dummy_env_name = config.env #config.run_name.split("_")[2]
    dummy_env = gym.make(dummy_env_name)

    env_obs_dim = eval_envs.observation_space.shape[0]
    env_act_dim = eval_envs.action_space.shape[0]



    # ghn_folder = os.path.join(*["data",actual_env_id, config.run_name, "models"])
    ghn_folder = config.path_to_ckpt
    probs_folder = os.path.join(*["checkpoints", "probs"])
    image_folder = os.path.join(*["checkpoints", "images"])
    os.makedirs(image_folder, exist_ok=True)
    rvc_folder = os.path.join(image_folder, "rvc")
    rvp_folder = os.path.join(image_folder, "rvp")
    os.makedirs(rvc_folder, exist_ok=True)
    os.makedirs(rvp_folder, exist_ok=True)
    model_dict_folder = os.path.join(*["checkpoints", "model_dicts"])
    os.makedirs(model_dict_folder, exist_ok=True)
  

    print("\n\n")
    st = time.time()
    list_of_ckpts_seen = []

    # check if few logs exist
    list_of_existing_model_dicts = [os.path.join(model_dict_folder,file) for file in os.listdir(model_dict_folder)]
    if len(list_of_existing_model_dicts) > 0:
        print("Existing logs found for:")
        for model_dict_file in list_of_existing_model_dicts:
            epoch_num = model_dict_file.split("dict")[2].split(".")[0]
            existing_ckpt_file_name = os.path.join(ghn_folder,"ghn_" + epoch_num + ".pt")
            list_of_ckpts_seen.append(existing_ckpt_file_name)
            print("\t " + existing_ckpt_file_name)

    while True:

        # if (time.time() - st) > (config.hours * 60 * 60):
        #     break

        # list all models in the folder
        # list_of_ckpts = [os.path.join(ghn_folder,file) for file in os.listdir(ghn_folder)]
        list_of_ckpts = [ghn_folder]

        new_files = []
        for ckp_file in list_of_ckpts:
            if ckp_file in list_of_ckpts_seen:
                continue
            else:
                new_files.append(ckp_file)
        
        if len(new_files) > 0:
            print("New File(s) detected!!")
            # time.sleep(1)

        # sort new files
        # new_files.sort(key = lambda x: int(x.split("/")[-1].split(".")[0].split("_")[1]))
        # config = {}
        # config['max_shape'] = (512, 512, 1, 1)
        # config['num_classes'] = 4 * eval_envs.action_space.shape[0]
        # config['num_observations'] = eval_envs.observation_space.shape[0]
        # config['weight_norm'] = True
        # config['ve'] = 1 > 1
        # config['layernorm'] = True
        # config['hid'] = 16
        action_space = eval_envs.action_space
        num_inputs = eval_envs.observation_space.shape[0]
        for ckp_file in new_files:
            print(f"Evaluating {ckp_file}")
            # epoch_num = int(ckp_file.split("/")[-1].split(".")[0].split("_")[1])
            epoch_num = 5000
            # prob_dict = torch.load(os.path.join(probs_folder, f"prob_{epoch_num}.pt"))
            # prob_dict = {f"layer{j+1}" : {layer_:prob_ for (layer_, prob_) in prob_dict[f"layer{j+1}"]} for j in range(4)}
            actor = hyperActor(action_space.shape[0], num_inputs, action_space.high[0], np.arange(4,512 + 1))
            state_dict = torch.load(ckp_file, map_location='cpu')
            actor.load_state_dict(state_dict['ghn'])
            # ghn = MLP_GHN.load(ckp_file, debug_level=0, device='cpu', verbose=True, config = config)

            model_dict = get_model_dict(num_models_to_eval, list_of_allowable_layers, env_obs_dim, env_act_dim, eval_envs, actor.ghn)
            
            
            # evaluate(eval_envs, ghn, model_dict, list_of_arcs)

            # normalize reward in model_dict
            # model_dict['norm_reward'] = model_dict.apply (lambda row: dummy_env.get_normalized_score(row['reward']), axis=1).astype(np.float64)

            plot_model_dict(model_dict, actual_env_id, epoch_num, rvp_folder, rvc_folder)

            # save model_dict
            model_dict.to_csv(os.path.join(model_dict_folder, "model_dict" + str(epoch_num) +".csv"))

            print(f"Logging Done for file: {ckp_file} ")
            print(f"{time.time()-st} seconds have passed \n")

            list_of_ckpts_seen.append(ckp_file)
        
        time.sleep(1)


if __name__ == "__main__":
    config = get_config()

    main(config)
