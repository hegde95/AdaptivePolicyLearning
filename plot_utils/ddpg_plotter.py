import warnings

warnings.filterwarnings("ignore")
import os

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import argparse
import time
from itertools import product

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
from stable_baselines3.common.vec_env import SubprocVecEnv

from hyper.core import hyperActor
from hyper.ghn_modules import MLP_GHN, MlpNetwork
from DDPG.normalizer import Normalizer
from configs.config_helper import str2bool


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--hours", type=int, default=24, help="Hours to run the plotter script for")
    parser.add_argument("--run_dir", type=str, default="runs", help="Path to the base of all runs")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--num_evals", type=int, default=20, help="Number of evaluations to run per architecture, default: 20")
    parser.add_argument("--overwrite", type=str2bool, default="False", help="Over write the existing plots/model_dicts, default: False")
    
    parser.add_argument("--path_to_ckpt", type=str, help="Path to the ckpt file")

    parser.add_argument('--eval_every', type=int, default=1,
                        help='run eval every n epochs')
    parser.add_argument('--rand', type=str2bool, default="False",
                        help='run eval on random architectures')   

    parser.add_argument('--cuda', type=str2bool, default="False",
                        help='run on CUDA (default: False)')                                            
    parser.add_argument('--cuda_device', type=int, default=0,
                    help="sets the cuda device to run experiments on")
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


def evaluate_for_single_policy(env, model, state_normalizer, goal_normalizer, device = 'cpu'):
    success_vector = np.zeros(env.num_envs)
    average_forward_pass_time = 0
    env_dictionary = env.reset()
    # # decorrelate the envs with random actions
    # for _ in range(5):
    #     action = np.array([env.action_space.sample() for __ in range(env.num_envs)])
    #     dummy_obs,_,_,_ = env.step(action)

    env_dictionary = env.reset()
    s = env_dictionary["observation"]
    ag = env_dictionary["achieved_goal"]
    g = env_dictionary["desired_goal"]   
    while np.any([np.linalg.norm((ag - g)[i])<=0.05 for i in range(env.num_envs)]):
        env_dictionary = env.reset()
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]     
    # rewards = 0
    for t in range(50):
        # concatenate state and goal after normalizing
        s_ = state_normalizer.normalize(s)
        g_ = goal_normalizer.normalize(g)
        sg_ = np.concatenate([s_, g_], axis=-1)
        sg_ = torch.from_numpy(sg_).float().to(device)
        start_time = time.time()
        action = model(sg_)
        end_time = time.time()
        forward_pass_time = end_time - start_time
        average_forward_pass_time += forward_pass_time
        action = action[:,:action.shape[-1]//2]
        action = action.detach().cpu().numpy()
        observation_new, reward, done, info = env.step(action)
        s = observation_new['observation']
        g = observation_new['desired_goal']   
        success_vector = np.array([info__['is_success'] + success_vector[ind] for ind, info__ in enumerate(info)]).astype(bool)

        # rewards += reward

    success_rate = np.mean(success_vector) 
    average_forward_pass_time /= 50
    return success_rate, average_forward_pass_time

def cal_prob(fc_layer,prob_dict):
    model_prob = 1
    for j in range(4):
        if j < len(fc_layer):
            layer_prob = prob_dict[f"layer{j+1}"][fc_layer[j]]
        else:
            layer_prob = prob_dict[f"layer{j+1}"][0]
        model_prob *= layer_prob
    return model_prob

# def evaluate(env, ghn, model_dict, list_of_net_arch_eval): 
#     """
#     Makes an evaluation run with the current GHN
#     """
#     ghn.eval()
#     inp_dim = env.observation_space.shape[0]
#     out_dim = 2 *env.action_space.shape[0]
#     net_args = {
#         'fc_layers':[],
#         'inp_dim':inp_dim,
#         'out_dim':out_dim,
#     }      
#     hyper_reward = 0
#     for i,net_arch in tqdm.tqdm(enumerate(list_of_net_arch_eval)):
#         net_args['fc_layers'] = net_arch    
#         model = MlpNetwork(**net_args)
#         shape_ind = [[0]]
#         for layer in net_args['fc_layers']:
#             shape_ind.append([layer])
#             shape_ind.append([layer])
#         shape_ind.append([out_dim])
#         shape_ind.append([out_dim])    
#         shape_ind = torch.Tensor(shape_ind).to('cpu')          
#         ghn(model, shape_ind = shape_ind)  
#         reward_per_policy = evaluate_for_single_policy(env, model)

#         model_dict.at[model_dict.index[model_dict['architecture'] == str(net_args['fc_layers'])][0], 'reward'] = reward_per_policy        
#         hyper_reward += reward_per_policy

#     return hyper_reward/len(list_of_net_arch_eval)

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
    elif env_name in ["Humanoid-v2", "Humanoid-v3"]:
        return -30, 10000, 5, 13, 5, 15
    elif env_name in ["FetchPickAndPlace-v1"]:
        return -0.1, 1.1, 4, 10, 5, 14
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
    plt.scatter(np.log(model_dict['params']), model_dict['reward'], c=np.log(model_dict['exec_time']), s=10, cmap = 'coolwarm')
    plt.xlabel('log(num params)')
    plt.ylabel('reward')
    plt.xlim(p_low,p_high)
    plt.ylim(ylim_l, ylim_h)
    plt.title(f"{i} Epochs")
    plt.colorbar().set_label('log(exec_time)', rotation=270, labelpad = 10)
    plt.savefig(rvp_folder + '/' + str(i) + '.png', dpi = 300)
    plt.clf()

    model_dict = model_dict.sort_values(by = ['params'], ascending = True)

    plt.plot(bin_centers, bin_means)
    plt.scatter(np.log(model_dict['capacity']), model_dict['reward'], c=np.log(model_dict['exec_time']), s=10, cmap = 'coolwarm')
    plt.xlabel('log(capacity)')
    plt.ylabel('reward')
    plt.xlim(c_low,c_high)
    plt.ylim(ylim_l, ylim_h)
    plt.title(f"{i} Epochs")
    plt.colorbar().set_label('exec_time', rotation=270, labelpad = 10)
    # plt.plot(np.log(model_dict['capacity']), gaussian_filter1d(model_dict['norm_reward'], sigma =80))
    plt.savefig(rvc_folder + '/' + str(i) + '.png', dpi = 300)
    plt.clf()

def get_list_of_arcs(num_models_to_eval, list_of_allowable_layers, random_sequence = False):
    arcs = []
    if random_sequence:
        for i in range(num_models_to_eval):
            num_layer = np.random.choice([1,2,3,4])
            arc = list(np.random.choice(list_of_allowable_layers,num_layer))
            arcs.append(arc)
    else:
        for k in range(1,5):
            arcs.extend(list(product(list_of_allowable_layers, repeat = k)))
    return arcs

def get_model_dict(list_of_arcs, env_obs_dim, env_act_dim, env, ghn, state_normalizer, goal_normalizer, device = 'cpu'):

    inp_dim = env_obs_dim
    out_dim = 2 * env_act_dim
    net_args = {
        'fc_layers':[],
        'inp_dim':inp_dim,
        'out_dim':out_dim,
    }    
    model_dict = pd.DataFrame(columns=['architecture', 'reward', 'params', 'capacity', 'exec_time'])
    for arc in tqdm.tqdm(list_of_arcs):
        net_args['fc_layers'] = arc 
        model = MlpNetwork(**net_args)
        shape_ind = [[0]]
        for layer in net_args['fc_layers']:
            shape_ind.append([layer])
            shape_ind.append([layer])
        shape_ind.append([out_dim])
        shape_ind.append([out_dim])    
        shape_ind = torch.Tensor(shape_ind).to(device)          
        ghn(model, shape_ind = shape_ind)  
        reward_per_policy, average_forward_pass_time = evaluate_for_single_policy(env, model, state_normalizer, goal_normalizer, device)
        params = sum(p.numel() for p in model.parameters())
        cap = get_capacity(net_args['fc_layers'], inp_dim, out_dim)


        model_dict  = pd.concat([model_dict, pd.DataFrame({'architecture':[str(list(arc))],'reward':[reward_per_policy], 'params':[params], 'capacity':[cap], 'exec_time':[average_forward_pass_time] })], ignore_index=True)

    model_dict['params'] = model_dict['params'].astype(np.float64)
    model_dict['capacity'] = model_dict['capacity'].astype(np.float64)
    model_dict['reward'] = model_dict['reward'].astype(np.float64)
    model_dict['exec_time'] = model_dict['exec_time'].astype(np.float64)

    model_dict = model_dict.sort_values(by = ['params'], ascending= False)
    model_dict = model_dict.sort_values(by = ['capacity'], ascending = False)
    return model_dict


def main(config):
    num_models_to_eval = 10000
    if config.rand:
        list_of_allowable_layers = np.arange(4,512)
    else:
        list_of_allowable_layers = [4,8,16,32,64,128,256,512]
    actual_env_id = config.run_name.split('_')[3]
    largest_model = [list_of_allowable_layers[-1]]*4
    smallest_model = [list_of_allowable_layers[0]]
    baseline_model = [256,256]

    device = torch.device(f"cuda:{config.cuda_device}" if config.cuda else "cpu")

    eval_env_fns = [lambda: gym.make(actual_env_id) for _ in range(config.num_evals)]
    eval_envs = SubprocVecEnv(eval_env_fns)
    eval_envs.seed(config.seed)

    env_state_dim = eval_envs.observation_space['observation'].shape[0]
    env_goal_dim = eval_envs.observation_space['desired_goal'].shape[0]
    env_obs_dim = env_state_dim + env_goal_dim
    env_act_dim = eval_envs.action_space.shape[0]

    if config.path_to_ckpt is not None:
        path_to_ckpt = config.path_to_ckpt
        run_folder = os.path.join(*config.path_to_ckpt.split("/")[:-2])
    else:
        run_folder = os.path.join(*[config.run_dir, config.run_name])
        checkpoint_folder = os.path.join(*[run_folder, 'checkpoints'])
    image_folder = os.path.join(*[run_folder, "images_rand" if config.rand else "images"])
    os.makedirs(image_folder, exist_ok=True)
    rvc_folder = os.path.join(image_folder, "rvc")
    rvp_folder = os.path.join(image_folder, "rvp")
    os.makedirs(rvc_folder, exist_ok=True)
    os.makedirs(rvp_folder, exist_ok=True)
    model_dict_folder = os.path.join(*[run_folder, "model_dicts_rand" if config.rand else "model_dicts"])
    os.makedirs(model_dict_folder, exist_ok=True)
  

    print("\n\n")
    st = time.time()
    list_of_ckpts_seen = []

    report = pd.DataFrame(columns=['epoch','best_model','best_model_reward','largest_model','largest_model_reward','smallest_model','smallest_model_reward','baseline_model','baseline_model_reward'])

    # check if few logs exist
    list_of_existing_model_dicts = [os.path.join(model_dict_folder,file) for file in os.listdir(model_dict_folder)]
    list_of_existing_model_dicts.sort(key = lambda x: int(x.split("/")[-1].split(".")[0].split("dict")[-1]))
    if len(list_of_existing_model_dicts) > 0:
        print("Existing logs found for:")
        for model_dict_file in list_of_existing_model_dicts:
            epoch_num = model_dict_file.split("dict")[2].split(".")[0]
            existing_ckpt_file_name = os.path.join(checkpoint_folder,"sac_checkpoint_" + epoch_num)
            list_of_ckpts_seen.append(existing_ckpt_file_name)
            print("\t " + existing_ckpt_file_name)


    list_of_arcs = get_list_of_arcs(num_models_to_eval, list_of_allowable_layers, config.rand)

    while True:

        if (time.time() - st) > (config.hours * 60 * 60):
            break

        # list all models in the folder
        list_of_ckpts = [os.path.join(checkpoint_folder,file) for file in os.listdir(checkpoint_folder)]
        list_of_ckpts_to_eval = []

        list_of_ckpts.sort(key = lambda x: int(x.split("/")[-1].split(".")[0].split("_")[2]))
        # select every args.eval_every checkpoints
        for j in range(0, len(list_of_ckpts), config.eval_every):
            list_of_ckpts_to_eval.append(list_of_ckpts[j])


        new_files = []
        for ckp_file in list_of_ckpts_to_eval:
            if (ckp_file in list_of_ckpts_seen) and not config.overwrite:
                continue
            else:
                new_files.append(ckp_file)

        if len(new_files) > 0:
            if config.overwrite:
                print("Overwriting existing files")
            else:
                print("New File(s) detected!!")
            # time.sleep(1)

        # sort new files
        new_files.sort(key = lambda x: int(x.split("/")[-1].split(".")[0].split("_")[2]), reverse=True)

        action_space = eval_envs.action_space
        num_inputs = env_obs_dim
        for ckp_file in new_files:
            print(f"Evaluating {ckp_file}")
            # sleep for a bit to make sure the ckpt is saved
            time.sleep(1) #10
            epoch_num = int(ckp_file.split("/")[-1].split(".")[0].split("_")[2])

            actor = hyperActor(action_space.shape[0], num_inputs, action_space.high[0], np.array([4,8,16,32,64,128,256,512]), device=device)
            state_dict = torch.load(ckp_file, map_location=device)
            state_dict['policy_state_dict'].pop('conditional_arc_dist.0.weight', None)
            state_dict['policy_state_dict'].pop('conditional_arc_dist.0.bias', None)
            state_dict['policy_state_dict'].pop('conditional_arc_dist.2.weight', None)
            state_dict['policy_state_dict'].pop('conditional_arc_dist.2.bias', None)
            actor.load_state_dict(state_dict['policy_state_dict'])

            state_normalizer = Normalizer(env_state_dim, default_clip_range=5)
            goal_normalizer = Normalizer(env_goal_dim, default_clip_range=5)
            state_normalizer.mean = state_dict['state_normalizer_mean']
            state_normalizer.std = state_dict['state_normalizer_std']
            goal_normalizer.mean = state_dict['goal_normalizer_mean']
            goal_normalizer.std = state_dict['goal_normalizer_std']

            np.random.shuffle(list_of_arcs)
            model_dict = get_model_dict(list_of_arcs, env_obs_dim, env_act_dim, eval_envs, actor.ghn, state_normalizer = state_normalizer, goal_normalizer = goal_normalizer, device = device)
            
            # reward_largest_model = model_dict[model_dict.architecture == str(largest_model)].reward.item()
            # reward_smallest_model = model_dict[model_dict.architecture == str(smallest_model)].reward.item()
            # reward_baseline_model = model_dict[model_dict.architecture == str(baseline_model)].reward.item()

            # reward_best_model = model_dict.reward.max()
            # best_model = model_dict[model_dict.reward == reward_best_model].architecture.item()

            # report  = pd.concat([report, pd.DataFrame({'epoch': [epoch_num], 'best_model': [best_model], 'best_model_reward': [reward_best_model], 'largest_model': [str(largest_model)], 'largest_model_reward': [reward_largest_model], 'smallest_model': [str(smallest_model)], 'smallest_model_reward': [reward_smallest_model], 'baseline_model': [str(baseline_model)], 'baseline_model_reward': [reward_baseline_model]})], ignore_index=True)
            # report.to_csv(os.path.join(run_folder, "report.csv"), index=False)

            plot_model_dict(model_dict, actual_env_id, epoch_num, rvp_folder, rvc_folder)

            # save model_dict
            model_dict = model_dict.sort_values(by = ['params'], ascending = True)            
            model_dict.to_csv(os.path.join(model_dict_folder, "model_dict" + str(epoch_num) +".csv"))

            print(f"Logging Done for file: {ckp_file} ")
            print(f"{time.time()-st} seconds have passed \n")

            list_of_ckpts_seen.append(ckp_file)

        # time.sleep(1)
        # print("Done")


if __name__ == "__main__":
    config = get_config()

    main(config)
