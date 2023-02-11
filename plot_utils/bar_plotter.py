import argparse
from pickle import TRUE
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from copy import deepcopy as dc
from plot_utils.sac_plotter import get_capacity

def get_config():
    parser = argparse.ArgumentParser(description='Plot archwise plot accross epochs')
    # parser.add_argument('--runs', type=str, help='Comma separated list of runs to plot')
    parser.add_argument('--run_dir', default = "runs", type=str, help='Directory where the runs are stored')
    parser.add_argument('--dest', default="./figs/", type=str, help='Destination folder to save plots')
    # parser.add_argument('--prefix', default="", type=str, help='Prefix to add to the plot name')

    return parser.parse_args()

def get_nearest_epochs(epoch, epochs_list):
    # will return the nearest smaller and nearest greater epochs
    smaller_epoch = None
    greater_epoch = None
    if epoch in epochs_list:
        return epoch, epoch
    elif epoch < min(epochs_list):
        return min(epochs_list),min(epochs_list)
    elif epoch > max(epochs_list):
        return max(epochs_list),max(epochs_list)
    else:
        for e in epochs_list:
            if e > epoch:
                break
            smaller_epoch = e
        epochs_list.reverse()
        for e in epochs_list:                
            if e < epoch:
                break
            greater_epoch = e
        return smaller_epoch, greater_epoch

def linear_interpolation(x, x_1, x_2, y_1, y_2):
    return y_1 + (x - x_1) * (y_2 - y_1) / (x_2 - x_1)

def is_valid_arc(arc_str, min_layer_size = 4):
    arc = [int(l) for l in arc_str.strip(']').strip('[').split(',')]
    if min_layer_size >= min(arc):
        return False
    return np.prod([arc[i] >= arc[i+1] for i in range(len(arc) - 1)], dtype=bool)


def main(config):
    list_of_runs = [os.path.join(*[config.run_dir, run]) for run in config.runs.split(',')]

    # list all model_dicts from these runs
    list_of_path_to_model_dicts = [os.path.join(*[run, 'model_dicts']) for run in list_of_runs]

    # list of seeds
    list_of_seeds = [int(run.split("_")[-1]) for run in list_of_runs]
    list_of_epochs = [
        # 0,
        80000,
        400000,
        720000,
        1040000,
        1360000,
        1680000,
        2000000,
        2320000,
        2640000,
        2960000,
    ] 

    TICK_SIZE = 100
    LABEL_SIZE = 120
    LEGEND_SIZE = 100

    CAPACITY = False
    REWARD = True
    PARAM = True

    LINEWITDTH = 8

    reward_bars = {"row_max":1, 
                # "row_99_percent":0.99, 
                "row_90_percent":0.9, 
                "row_80_percent":0.8, 
                # "row_70_percent":0.7, 
                # "row_60_percent":0.6, 
                # "row_50_percent":0.5, 
                # "row_40_percent":0.4, 
                # "row_30_percent":0.3, 
                # "row_20_percent":0.2, 
                # "row_10_percent":0.1
                }

    plot_bars = {"row_max":"max", 
                # "row_99_percent":"99%", 
                "row_90_percent":"90-95%", 
                "row_80_percent":"80-85%", 
                # "row_60_percent":"60-65%", 
                # "row_40_percent":"0.4%", 
                # "row_20_percent":"0.2%", 
                }

    REWARD_MIN_MAX = False
    PARAMS_MIN_MAX = True

    baseline_data_folder = os.path.join(*["baseline_data", config.prefix])
    baseline_rewards_dict = {epoch: { seed: None for seed in [0,1,2,3,4]} for epoch in list_of_epochs}
    for s,file in enumerate(os.listdir(baseline_data_folder)):
        path_to_file = os.path.join(*[baseline_data_folder, file])
        baseline_data = pd.read_csv(path_to_file)
        for epoch in list_of_epochs:
            # get the nearest smaller and greater epochs
            smaller_epoch, greater_epoch = get_nearest_epochs(epoch, dc(baseline_data[baseline_data.columns[0]].tolist()))
            # get the corresponding rewards
            smaller_epoch_reward = baseline_data[baseline_data.columns[1]][baseline_data[baseline_data.columns[0]] == smaller_epoch].tolist()[0]
            greater_epoch_reward = baseline_data[baseline_data.columns[1]][baseline_data[baseline_data.columns[0]] == greater_epoch].tolist()[0]
            if greater_epoch_reward != smaller_epoch_reward:
                # interpolate the reward
                interpolated_reward = linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_epoch_reward, greater_epoch_reward)
            else:
                interpolated_reward = greater_epoch_reward
            baseline_rewards_dict[epoch][s] = interpolated_reward

    baseline_reward_means = np.array([np.mean([baseline_rewards_dict[epoch][s] for s in baseline_rewards_dict[epoch]]) for epoch in baseline_rewards_dict])
    baseline_reward_means_dict = {epoch: np.mean([baseline_rewards_dict[epoch][s] for s in baseline_rewards_dict[epoch]]) for epoch in baseline_rewards_dict}
    baseline_reward_stds = np.array([np.std([baseline_rewards_dict[epoch][s] for s in baseline_rewards_dict[epoch]]) for epoch in baseline_rewards_dict])


    # lambda fn to get baseline number of params given inp, out
    get_baseline_params = lambda inp, out: ((inp + 1) * 256) + ((256 + 1) * 256) + ((256 + 1) * out)
    # inp out dict per env
    inp_out_dict = {
        "humanoid":(376, 17),
        "ant":(111, 8),
        "hc":(17,6),
        "hopper":(11,3),
        "walker":(17,6)
    }

    baseline_params = get_baseline_params(*inp_out_dict[config.prefix])

    baseline_capacity = get_capacity([256,256], inp_out_dict[config.prefix][0], inp_out_dict[config.prefix][1])

    reward_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    param_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    capacity_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    for path_to_model_dicts, seed in zip(list_of_path_to_model_dicts, list_of_seeds):
        list_of_model_dicts = os.listdir(path_to_model_dicts)
        list_of_model_dicts.sort(key = lambda x:int(x.split("dict")[1].split(".")[0]))
        list_of_model_dicts_epochs = [int(model_dict.split("dict")[1].split(".")[0]) for model_dict in list_of_model_dicts]
        

        tmp_rewards_dict = {}
        tmp_params_dict = {}
        tmp_capacity_dict = {}
        # get model_dicts closest to the list of epochs
        list_of_model_dicts_to_use = []
        for epoch in list_of_epochs:
            # get nearest epochs
            (smaller_epoch, greater_epoch) = get_nearest_epochs(epoch, dc(list_of_model_dicts_epochs))
            # get the corresponding model_dicts file names
            smaller_epoch_model_dict = f"model_dict{smaller_epoch}.csv"
            greater_epoch_model_dict = f"model_dict{greater_epoch}.csv"
            # add to list of model_dicts to use
            if not smaller_epoch_model_dict in list_of_model_dicts_to_use:
                list_of_model_dicts_to_use.append(smaller_epoch_model_dict)
            if not greater_epoch_model_dict in list_of_model_dicts_to_use:
                list_of_model_dicts_to_use.append(greater_epoch_model_dict)
            # continue

        for model_dict in list_of_model_dicts_to_use:
            epoch = int(model_dict.split("dict")[1].split(".")[0])
            
            # if epoch in list_of_epochs:
            model_dict_path = os.path.join(path_to_model_dicts, model_dict)
            # read csv file
            df = pd.read_csv(model_dict_path)

            # remove bad arcs
            # df = df[df.apply(lambda x: is_valid_arc(x.architecture, 16), axis = 1)]
            df = df[df.reward > 0]
            if len(df) == 0:
                continue
            
            # create a normalized reward column
            df['norm_reward'] = (df['reward'] - df['reward'].min()) / (df['reward'].max() - df['reward'].min())  
            tmp_rewards_dict[epoch] = {}
            tmp_params_dict[epoch] = {} 
            tmp_capacity_dict[epoch] = {} 
            
            max_reward = df.norm_reward.max()

            # closest to baseline epoch reward
            greater_epoch, smaller_epoch = get_nearest_epochs(epoch, dc(list_of_epochs))
            if greater_epoch == smaller_epoch:
                interpolated_baseline_reward = baseline_reward_means_dict[greater_epoch]
            else:
                greater_baseline_reward = baseline_reward_means_dict[greater_epoch]
                smaller_baseline_reward = baseline_reward_means_dict[smaller_epoch]
                interpolated_baseline_reward = linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_baseline_reward, greater_baseline_reward)

            # max_reward = (interpolated_baseline_reward - df['reward'].min())/ (df['reward'].max() - df['reward'].min())

            
          
            for bar_key in reward_bars.keys():
                # if bar_key == 'row_max':
                high_performing_archs = df[df.norm_reward >= reward_bars[bar_key]*max_reward]
                # else:
                #     high_performing_archs =  df[(df.norm_reward >= reward_bars[bar_key]*df.norm_reward.max()) & (df.norm_reward <= ((reward_bars[bar_key] + 0.05)*df.norm_reward.max()))]

                # high_performing_archs = high_performing_archs[high_performing_archs.reward == high_performing_archs.reward.min()]

                # To atleast get this much reward:
                smallest_reward = min(high_performing_archs.reward) 
                # we need atleast this many parameters
                smallest_param = min(high_performing_archs.params)

                # find smallest model that at least achieves smallest_reward
                tmp = high_performing_archs[high_performing_archs.params == smallest_param]

                # among two model that have same number of models,
                # choose the one with higher reward
                # tmp = tmp[tmp.capacity == tmp.capacity.max()]                
                tmp = tmp[tmp.norm_reward == tmp.norm_reward.max()]

                tmp_rewards_dict[epoch][bar_key] = tmp.reward.item()
                tmp_params_dict[epoch][bar_key] = tmp.params.item()
                tmp_capacity_dict[epoch][bar_key] = tmp.capacity.item()



        # find values for each epoch in list of all epochs
        for epoch in list_of_epochs:
            # get nearest epoch in tmp_rewards_dict.keys()
            smaller_epoch, greater_epoch = get_nearest_epochs(epoch, dc(list(tmp_rewards_dict.keys())))
            # get the values for the nearest epochs
            smaller_epoch_reward = tmp_rewards_dict[smaller_epoch]
            greater_epoch_reward = tmp_rewards_dict[greater_epoch]
            # get the values for the nearest epochs
            smaller_epoch_param = tmp_params_dict[smaller_epoch]
            greater_epoch_param = tmp_params_dict[greater_epoch]
            # get the values for the nearest epochs
            smaller_epoch_capacity = tmp_capacity_dict[smaller_epoch]
            greater_epoch_capacity = tmp_capacity_dict[greater_epoch]
            

            if smaller_epoch == greater_epoch:
                reward_results_dict[epoch][seed] = smaller_epoch_reward
                param_results_dict[epoch][seed] = smaller_epoch_param
                capacity_results_dict[epoch][seed] = smaller_epoch_capacity
            else:
                # interpolate the values for the nearest epochs
                reward_results_dict[epoch][seed] = {key:linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_epoch_reward[key], greater_epoch_reward[key]) for key in reward_bars.keys()}

                param_results_dict[epoch][seed] = {key:linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_epoch_param[key], greater_epoch_param[key]) for key in reward_bars.keys()}

                capacity_results_dict[epoch][seed] = {key:linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_epoch_capacity[key], greater_epoch_capacity[key]) for key in reward_bars.keys()}



    # average and std across all seeds
    average_rewards_dict = {}
    average_params_dict = {}
    average_capacity_dict = {}
    std_rewards_dict = {}
    std_params_dict = {}
    std_capacity_dict = {}
    max_rewards_dict = {}
    min_rewards_dict = {}
    min_capacity_dict = {}
    min_params_dict = {}
    max_params_dict = {}
    max_capacity_dict = {}
    for epoch in list_of_epochs:
        average_rewards_dict[epoch] = {
            bar_key: np.mean([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        average_params_dict[epoch] = {
            bar_key: np.mean([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        average_capacity_dict[epoch] = {
            bar_key: np.mean([capacity_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }        
        std_rewards_dict[epoch] = {
            bar_key: np.std([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        std_params_dict[epoch] = {
            bar_key: np.std([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        std_capacity_dict[epoch] = {
            bar_key: np.std([capacity_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }        
        max_rewards_dict[epoch] = {
            bar_key: np.max([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        max_params_dict[epoch] = {
            bar_key: np.max([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        max_capacity_dict[epoch] = {
            bar_key: np.max([capacity_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }        
        min_rewards_dict[epoch] = {
            bar_key: np.min([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        min_params_dict[epoch] = {
            bar_key: np.min([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        min_capacity_dict[epoch] = {
            bar_key: np.min([capacity_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }        

    # subplot average and std rewards and params
    if CAPACITY:
        fig, axs = plt.subplots(1,3, figsize=(30, 15))
    else:
        fig, axs = plt.subplots(1,2, figsize=(48, 28))

    for plot_key in plot_bars.keys():
        axs[0].plot(list_of_epochs, [average_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], label=plot_bars[plot_key], linewidth= LINEWITDTH)

    if REWARD_MIN_MAX:
        for plot_key in plot_bars.keys():
            axs[0].fill_between(list_of_epochs, [min_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], [max_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)
    else:
        for plot_key in plot_bars.keys():
            axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch][plot_key] - std_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], [average_rewards_dict[epoch][plot_key] + std_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)

    # add baseline data
    axs[0].plot(list_of_epochs, baseline_reward_means, label='baseline', linewidth= LINEWITDTH)
    axs[0].fill_between(list_of_epochs, baseline_reward_means - baseline_reward_stds, baseline_reward_means + baseline_reward_stds, alpha=0.2)

    axs[0].set_xlabel("steps", fontsize = LABEL_SIZE)
    axs[0].set_ylabel("reward", fontsize = LABEL_SIZE)
    # set x ticks to every 1000000
    axs[0].set_xticks([0, 1000000, 2000000, 3000000])
    # increase label font size
    axs[0].tick_params(axis='both', labelsize=TICK_SIZE)
    # increase xlabel font size
    axs[0].xaxis.label.set_size(LABEL_SIZE)
    # increase ylabel font size
    axs[0].yaxis.label.set_size(LABEL_SIZE)
    axs[0].xaxis.offsetText.set_fontsize(TICK_SIZE)

    # axs[0].legend()

    # add some space between subplots
    fig.subplots_adjust(wspace=0.4)
    

    for plot_key in plot_bars.keys():
        axs[1].plot(list_of_epochs, [average_params_dict[epoch][plot_key] for epoch in list_of_epochs], label=plot_bars[plot_key], linewidth= LINEWITDTH)

    if PARAMS_MIN_MAX:
        for plot_key in plot_bars.keys():
            axs[1].fill_between(list_of_epochs, [min_params_dict[epoch][plot_key] for epoch in list_of_epochs], [max_params_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)
    else:
        for plot_key in plot_bars.keys():
            axs[1].fill_between(list_of_epochs, [average_params_dict[epoch][plot_key] - std_params_dict[epoch][plot_key] for epoch in list_of_epochs], [average_params_dict[epoch][plot_key] + std_params_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)

    # add baseline data as a black line at baseline_params
    axs[1].plot(list_of_epochs, [baseline_params for epoch in list_of_epochs], label='baseline',linewidth = LINEWITDTH)


    # make y axis logarithmic
    axs[1].set_yscale("log")

    axs[1].set_xlabel("steps", fontsize = LABEL_SIZE)
    axs[1].set_ylabel("log param", fontsize = LABEL_SIZE)
    # set x ticks to every 1000000
    axs[1].set_xticks([0, 1000000, 2000000, 3000000])
    # increase label font size
    axs[1].tick_params(axis='both', labelsize=TICK_SIZE)
    # increase xlabel font size
    axs[1].xaxis.label.set_size(LABEL_SIZE)
    # increase ylabel font size
    axs[1].yaxis.label.set_size(LABEL_SIZE)
    axs[1].xaxis.offsetText.set_fontsize(TICK_SIZE)

    # axs[1].legend()


    if CAPACITY:
        for plot_key in plot_bars.keys():
            axs[2].plot(list_of_epochs, [average_capacity_dict[epoch][plot_key] for epoch in list_of_epochs], label=plot_bars[plot_key],linewidth = LINEWITDTH)


        if PARAMS_MIN_MAX:
            for plot_key in plot_bars.keys():
                axs[2].fill_between(list_of_epochs, [min_capacity_dict[epoch][plot_key] for epoch in list_of_epochs], [max_capacity_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)
        else:
            for plot_key in plot_bars.keys():
                axs[2].fill_between(list_of_epochs, [average_capacity_dict[epoch][plot_key] - std_capacity_dict[epoch][plot_key] for epoch in list_of_epochs], [average_capacity_dict[epoch][plot_key] + std_capacity_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)

        # add baseline data as a black line at baseline_capacity
        axs[2].plot(list_of_epochs, [baseline_capacity for epoch in list_of_epochs], label='baseline',linewidth = LINEWITDTH)

        # make y axis logarithmic
        axs[2].set_yscale("log")

        axs[2].set_xlabel("steps", fontsize = LABEL_SIZE)
        axs[2].set_ylabel("log capacity", fontsize = LABEL_SIZE)
        # increase label font size
        axs[2].tick_params(axis='both', labelsize=TICK_SIZE)
        # increase xlabel font size
        axs[2].xaxis.label.set_size(LABEL_SIZE)
        # increase ylabel font size
        axs[2].yaxis.label.set_size(LABEL_SIZE)
        axs[2].xaxis.offsetText.set_fontsize(TICK_SIZE)
        

        # axs[2].legend()

    lines, labels = axs[-1].get_legend_handles_labels()


    # add legend to the whole figure
    if CAPACITY:
        n_leg = 5
    else:
        n_leg = 2
    fig.legend(lines, labels, loc = 'upper center', ncol=n_leg, fontsize=LEGEND_SIZE)
    os.makedirs(config.dest, exist_ok=True)
    fig.savefig(os.path.join(config.dest, f"{config.prefix}_average_rewards_and_params.png"))
    
    
    pass

    
if __name__ == "__main__":
    config = get_config()
    dict_of_envs = {
        "hc": "2022-07-25_01-05-01_SAC_HalfCheetah-v2_Gaussian__hyper_111,2022-07-25_01-05-13_SAC_HalfCheetah-v2_Gaussian__hyper_222,2022-07-25_01-05-21_SAC_HalfCheetah-v2_Gaussian__hyper_333,2022-07-25_01-05-28_SAC_HalfCheetah-v2_Gaussian__hyper_444,2022-07-25_01-05-36_SAC_HalfCheetah-v2_Gaussian__hyper_555,2022-07-25_01-05-41_SAC_HalfCheetah-v2_Gaussian__hyper_666",
        "ant":"2022-08-01_01-30-38_SAC_Ant-v2_Gaussian__hyper_111,2022-08-01_01-30-46_SAC_Ant-v2_Gaussian__hyper_222,2022-08-01_01-30-55_SAC_Ant-v2_Gaussian__hyper_333,2022-08-01_01-31-03_SAC_Ant-v2_Gaussian__hyper_444,2022-08-01_01-31-10_SAC_Ant-v2_Gaussian__hyper_555",
        "walker":"2022-08-14_03-28-34_SAC_Walker2d-v2_Gaussian__hyper_111,2022-08-14_03-28-49_SAC_Walker2d-v2_Gaussian__hyper_222,2022-08-14_03-29-11_SAC_Walker2d-v2_Gaussian__hyper_333,2022-08-14_03-29-26_SAC_Walker2d-v2_Gaussian__hyper_444,2022-08-14_03-29-40_SAC_Walker2d-v2_Gaussian__hyper_555",
        "hopper":"2022-08-02_07-00-54_SAC_Hopper-v2_Gaussian__hyper_111,2022-08-02_07-01-17_SAC_Hopper-v2_Gaussian__hyper_222,2022-08-02_07-01-32_SAC_Hopper-v2_Gaussian__hyper_333,2022-08-02_07-01-48_SAC_Hopper-v2_Gaussian__hyper_444,2022-08-02_07-02-04_SAC_Hopper-v2_Gaussian__hyper_555",
        "humanoid":"2022-08-18_17-31-43_SAC_Humanoid-v2_Gaussian__hyper_111,2022-08-18_17-32-00_SAC_Humanoid-v2_Gaussian__hyper_222,2022-08-18_17-32-10_SAC_Humanoid-v2_Gaussian__hyper_333,2022-08-18_17-32-20_SAC_Humanoid-v2_Gaussian__hyper_444,2022-08-18_17-32-31_SAC_Humanoid-v2_Gaussian__hyper_555",
    }
    for env in dict_of_envs.keys():
        config.prefix = env
        config.runs = dict_of_envs[env]
        print(f"running for {env}")
        main(config)
