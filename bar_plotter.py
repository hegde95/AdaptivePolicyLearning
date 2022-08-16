import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def get_config():
    parser = argparse.ArgumentParser(description='Plot archwise plot accross epochs')
    parser.add_argument('--runs', type=str, help='Comma separated list of runs to plot')
    parser.add_argument('--run_dir', default = "runs", type=str, help='Directory where the runs are stored')
    parser.add_argument('--dest', default="./figs/", type=str, help='Destination folder to save plots')
    parser.add_argument('--prefix', default="", type=str, help='Prefix to add to the plot name')

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

    reward_bars = {"row_max":1, "row_99_percent":0.99, "row_90_percent":0.9, "row_80_percent":0.8, "row_70_percent":0.7, "row_60_percent":0.6, "row_50_percent":0.5, "row_40_percent":0.4, "row_30_percent":0.3, "row_20_percent":0.2, "row_10_percent":0.1}

    plot_bars = {"row_max":"max", 
                "row_99_percent":"99%", 
                "row_90_percent":"0.9%", 
                "row_80_percent":"0.8%", 
                "row_60_percent":"0.6%", 
                "row_40_percent":"0.4%", 
                "row_20_percent":"0.2%", 
                }

    REWARD_MIN_MAX = False
    PARAMS_MIN_MAX = True

    reward_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    param_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    for path_to_model_dicts, seed in zip(list_of_path_to_model_dicts, list_of_seeds):
        list_of_model_dicts = os.listdir(path_to_model_dicts)
        list_of_model_dicts.sort(key = lambda x:int(x.split("dict")[1].split(".")[0]))

        tmp_rewards_dict = {}
        tmp_params_dict = {}
        for model_dict in list_of_model_dicts:
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
            
          
            for bar_key in reward_bars.keys():
                high_performing_archs = df[df.norm_reward >= reward_bars[bar_key]*df.norm_reward.max()]
                smallest_param = min(high_performing_archs.params)
                tmp = high_performing_archs[high_performing_archs.params == smallest_param]
                tmp = tmp[tmp.capacity == tmp.capacity.max()]                
                tmp = tmp[tmp.norm_reward == tmp.norm_reward.max()]
                tmp_rewards_dict[epoch][bar_key] = tmp.reward.item()
                tmp_params_dict[epoch][bar_key] = tmp.params.item()



        # find values for each epoch in list of all epochs
        for epoch in list_of_epochs:
            # get nearest epoch in tmp_rewards_dict.keys()
            smaller_epoch, greater_epoch = get_nearest_epochs(epoch, list(tmp_rewards_dict.keys()))
            # get the values for the nearest epochs
            smaller_epoch_reward = tmp_rewards_dict[smaller_epoch]
            greater_epoch_reward = tmp_rewards_dict[greater_epoch]
            # get the values for the nearest epochs
            smaller_epoch_param = tmp_params_dict[smaller_epoch]
            greater_epoch_param = tmp_params_dict[greater_epoch]

            if smaller_epoch == greater_epoch:
                reward_results_dict[epoch][seed] = smaller_epoch_reward
                param_results_dict[epoch][seed] = smaller_epoch_param
            else:
                # interpolate the values for the nearest epochs
                reward_results_dict[epoch][seed] = {key:linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_epoch_reward[key], greater_epoch_reward[key]) for key in reward_bars.keys()}

                param_results_dict[epoch][seed] = {key:linear_interpolation(epoch, smaller_epoch, greater_epoch, smaller_epoch_param[key], greater_epoch_param[key]) for key in reward_bars.keys()}




    # average and std across all seeds
    average_rewards_dict = {}
    average_params_dict = {}
    std_rewards_dict = {}
    std_params_dict = {}
    max_rewards_dict = {}
    min_rewards_dict = {}
    min_params_dict = {}
    max_params_dict = {}
    for epoch in list_of_epochs:
        average_rewards_dict[epoch] = {
            bar_key: np.mean([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        average_params_dict[epoch] = {
            bar_key: np.mean([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        std_rewards_dict[epoch] = {
            bar_key: np.std([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        std_params_dict[epoch] = {
            bar_key: np.std([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        max_rewards_dict[epoch] = {
            bar_key: np.max([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        min_rewards_dict[epoch] = {
            bar_key: np.min([reward_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        min_params_dict[epoch] = {
            bar_key: np.min([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }
        max_params_dict[epoch] = {
            bar_key: np.max([param_results_dict[epoch][seed][bar_key] for seed in list_of_seeds]) for bar_key in reward_bars.keys()
        }

    # subplot average and std rewards and params
    fig, axs = plt.subplots(1,2, figsize=(25, 15))

    for plot_key in plot_bars.keys():
        axs[0].plot(list_of_epochs, [average_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], label=plot_key)

    if REWARD_MIN_MAX:
        for plot_key in plot_bars.keys():
            axs[0].fill_between(list_of_epochs, [min_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], [max_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)
    else:
        for plot_key in plot_bars.keys():
            axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch][plot_key] - std_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], [average_rewards_dict[epoch][plot_key] + std_rewards_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)

    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("reward")
    axs[0].legend()

    

    for plot_key in plot_bars.keys():
        axs[1].plot(list_of_epochs, [average_params_dict[epoch][plot_key] for epoch in list_of_epochs], label=plot_key)

    if PARAMS_MIN_MAX:
        for plot_key in plot_bars.keys():
            axs[1].fill_between(list_of_epochs, [min_params_dict[epoch][plot_key] for epoch in list_of_epochs], [max_params_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)
    else:
        for plot_key in plot_bars.keys():
            axs[1].fill_between(list_of_epochs, [average_params_dict[epoch][plot_key] - std_params_dict[epoch][plot_key] for epoch in list_of_epochs], [average_params_dict[epoch][plot_key] + std_params_dict[epoch][plot_key] for epoch in list_of_epochs], alpha=0.2)



    # make y axis logarithmic
    axs[1].set_yscale("log")

    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("log param")
    axs[1].legend()

    os.makedirs(config.dest, exist_ok=True)
    fig.savefig(os.path.join(config.dest, f"{config.prefix}_average_rewards_and_params.png"))
    
    
    pass

    
if __name__ == "__main__":
    config = get_config()
    main(config)