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

    return parser.parse_args()

def main(config):
    list_of_runs = [os.path.join(*[config.run_dir, run]) for run in config.runs.split(',')]

    # list all model_dicts from these runs
    list_of_path_to_model_dicts = [os.path.join(*[run, 'model_dicts']) for run in list_of_runs]

    # list of seeds
    list_of_seeds = [int(run.split("_")[-1]) for run in list_of_runs]
    list_of_epochs = []
    # for path_to_model_dicts, seed in zip(list_of_path_to_model_dicts, list_of_seeds):
    #     list_of_model_dicts = os.listdir(path_to_model_dicts)
    #     list_of_model_dicts.sort(key = lambda x:int(x.split("dict")[1].split(".")[0]))
    #     for model_dict in list_of_model_dicts:
    #         epoch = int(model_dict.split("dict")[1].split(".")[0]) 
    #         if epoch not in list_of_epochs:
    #             list_of_epochs.append(epoch)  
    list_of_epochs = [
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
        3280000,
        # 3600000,
        # 3920000,
        # 4240000,
        # 4560000,
        # 4880000,
        # 5200000,
        # 5520000,
        # 5840000,
        # 6160000,
        # 6480000,
        # 6800000,
        # 7120000,
        # 7440000,
        # 7760000,

    ] 

    reward_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    param_results_dict = { epoch: {seed: [] for seed in list_of_seeds} for epoch in list_of_epochs}

    for path_to_model_dicts, seed in zip(list_of_path_to_model_dicts, list_of_seeds):
        list_of_model_dicts = os.listdir(path_to_model_dicts)
        list_of_model_dicts.sort(key = lambda x:int(x.split("dict")[1].split(".")[0]))
        for model_dict in list_of_model_dicts:
            epoch = int(model_dict.split("dict")[1].split(".")[0])
            
            if epoch in list_of_epochs:
                model_dict_path = os.path.join(path_to_model_dicts, model_dict)
                # read csv file
                df = pd.read_csv(model_dict_path)
                max_row = df[df.reward == df.reward.max()]
                max_row_params = max_row.params

                # find all rows with 99% reward of the max_row
                high_performing_archs = df[df.reward >= 0.99*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_99_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_99_percent = row_99_percent[row_99_percent.capacity == row_99_percent.capacity.max()]                
                row_99_percent = row_99_percent[row_99_percent.reward == row_99_percent.reward.max()]


                # find all rows with 90% reward of the max_row
                high_performing_archs = df[df.reward >= 0.9*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_90_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_90_percent = row_90_percent[row_90_percent.capacity == row_90_percent.capacity.max()]
                row_90_percent = row_90_percent[row_90_percent.reward == row_90_percent.reward.max()]

                # find all rows with 80% reward of the max_row
                high_performing_archs = df[df.reward >= 0.8*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_80_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_80_percent = row_80_percent[row_80_percent.capacity == row_80_percent.capacity.max()]
                row_80_percent = row_80_percent[row_80_percent.reward == row_80_percent.reward.max()]

                # find all rows with 70% reward of the max_row
                high_performing_archs = df[df.reward >= 0.7*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_70_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_70_percent = row_70_percent[row_70_percent.capacity == row_70_percent.capacity.max()]
                row_70_percent = row_70_percent[row_70_percent.reward == row_70_percent.reward.max()]

                # find all rows with 60% reward of the max_row
                high_performing_archs = df[df.reward >= 0.6*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_60_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_60_percent = row_60_percent[row_60_percent.capacity == row_60_percent.capacity.max()]
                row_60_percent = row_60_percent[row_60_percent.reward == row_60_percent.reward.max()]

                # find all rows with 50% reward of the max_row
                high_performing_archs = df[df.reward >= 0.5*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_50_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_50_percent = row_50_percent[row_50_percent.capacity == row_50_percent.capacity.max()]
                row_50_percent = row_50_percent[row_50_percent.reward == row_50_percent.reward.max()]

                # find all rows with 40% reward of the max_row
                high_performing_archs = df[df.reward >= 0.4*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_40_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_40_percent = row_40_percent[row_40_percent.capacity == row_40_percent.capacity.max()]
                row_40_percent = row_40_percent[row_40_percent.reward == row_40_percent.reward.max()]

                # find all rows with 30% reward of the max_row
                high_performing_archs = df[df.reward >= 0.3*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_30_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_30_percent = row_30_percent[row_30_percent.capacity == row_30_percent.capacity.max()]
                row_30_percent = row_30_percent[row_30_percent.reward == row_30_percent.reward.max()]

                # find all rows with 20% reward of the max_row
                high_performing_archs = df[df.reward >= 0.2*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_20_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_20_percent = row_20_percent[row_20_percent.capacity == row_20_percent.capacity.max()]
                row_20_percent = row_20_percent[row_20_percent.reward == row_20_percent.reward.max()]

                # find all rows with 10% reward of the max_row
                high_performing_archs = df[df.reward >= 0.1*df.reward.max()]
                smallest_param = min(high_performing_archs.params)
                row_10_percent = high_performing_archs[high_performing_archs.params == smallest_param]
                row_10_percent = row_10_percent[row_10_percent.capacity == row_10_percent.capacity.max()]
                row_10_percent = row_10_percent[row_10_percent.reward == row_10_percent.reward.max()]

                reward_results_dict[epoch][seed] = {
                    "row_max":max_row.reward.item(),
                    "row_99_percent":row_99_percent.reward.item(),
                    "row_90_percent":row_90_percent.reward.item(),
                    "row_80_percent":row_80_percent.reward.item(), 
                    "row_70_percent":row_70_percent.reward.item(), 
                    "row_60_percent":row_60_percent.reward.item(), 
                    "row_50_percent":row_50_percent.reward.item(), 
                    "row_40_percent":row_40_percent.reward.item(), 
                    "row_30_percent":row_30_percent.reward.item(), 
                    "row_20_percent":row_20_percent.reward.item(), 
                    "row_10_percent":row_10_percent.reward.item(),                 
                }
                param_results_dict[epoch][seed] = {
                    "row_max": max_row.params.item(),
                    "row_99_percent": row_99_percent.params.item(),
                    "row_90_percent": row_90_percent.params.item(),
                    "row_80_percent": row_80_percent.params.item(),
                    "row_70_percent": row_70_percent.params.item(),
                    "row_60_percent": row_60_percent.params.item(),
                    "row_50_percent": row_50_percent.params.item(),
                    "row_40_percent": row_40_percent.params.item(),
                    "row_30_percent": row_30_percent.params.item(),
                    "row_20_percent": row_20_percent.params.item(),
                    "row_10_percent": row_10_percent.params.item()
                }

    # average and std across all seeds
    average_rewards_dict = {}
    average_params_dict = {}
    std_rewards_dict = {}
    std_params_dict = {}
    min_params_dict = {}
    max_params_dict = {}
    for epoch in list_of_epochs:
        average_rewards_dict[epoch] = {
            "row_max": np.mean([reward_results_dict[epoch][seed]["row_max"] for seed in list_of_seeds]),
            "row_99_percent": np.mean([reward_results_dict[epoch][seed]["row_99_percent"] for seed in list_of_seeds]),
            "row_90_percent": np.mean([reward_results_dict[epoch][seed]["row_90_percent"] for seed in list_of_seeds]),
            "row_80_percent": np.mean([reward_results_dict[epoch][seed]["row_80_percent"] for seed in list_of_seeds]),
            "row_70_percent": np.mean([reward_results_dict[epoch][seed]["row_70_percent"] for seed in list_of_seeds]),
            "row_60_percent": np.mean([reward_results_dict[epoch][seed]["row_60_percent"] for seed in list_of_seeds]),
            "row_50_percent": np.mean([reward_results_dict[epoch][seed]["row_50_percent"] for seed in list_of_seeds]),
            "row_40_percent": np.mean([reward_results_dict[epoch][seed]["row_40_percent"] for seed in list_of_seeds]),
            "row_30_percent": np.mean([reward_results_dict[epoch][seed]["row_30_percent"] for seed in list_of_seeds]),
            "row_20_percent": np.mean([reward_results_dict[epoch][seed]["row_20_percent"] for seed in list_of_seeds]),
            "row_10_percent": np.mean([reward_results_dict[epoch][seed]["row_10_percent"] for seed in list_of_seeds]),
        }

        average_params_dict[epoch] = {
            "row_max": np.mean([param_results_dict[epoch][seed]["row_max"] for seed in list_of_seeds]),
            "row_99_percent": np.mean([param_results_dict[epoch][seed]["row_99_percent"] for seed in list_of_seeds]),
            "row_90_percent": np.mean([param_results_dict[epoch][seed]["row_90_percent"] for seed in list_of_seeds]),
            "row_80_percent": np.mean([param_results_dict[epoch][seed]["row_80_percent"] for seed in list_of_seeds]),
            "row_70_percent": np.mean([param_results_dict[epoch][seed]["row_70_percent"] for seed in list_of_seeds]),
            "row_60_percent": np.mean([param_results_dict[epoch][seed]["row_60_percent"] for seed in list_of_seeds]),
            "row_50_percent": np.mean([param_results_dict[epoch][seed]["row_50_percent"] for seed in list_of_seeds]),
            "row_40_percent": np.mean([param_results_dict[epoch][seed]["row_40_percent"] for seed in list_of_seeds]),
            "row_30_percent": np.mean([param_results_dict[epoch][seed]["row_30_percent"] for seed in list_of_seeds]),
            "row_20_percent": np.mean([param_results_dict[epoch][seed]["row_20_percent"] for seed in list_of_seeds]),
            "row_10_percent": np.mean([param_results_dict[epoch][seed]["row_10_percent"] for seed in list_of_seeds]),
        }

        std_rewards_dict[epoch] = {
            "row_max": np.std([reward_results_dict[epoch][seed]["row_max"] for seed in list_of_seeds]),
            "row_99_percent": np.std([reward_results_dict[epoch][seed]["row_99_percent"] for seed in list_of_seeds]),
            "row_90_percent": np.std([reward_results_dict[epoch][seed]["row_90_percent"] for seed in list_of_seeds]),
            "row_80_percent": np.std([reward_results_dict[epoch][seed]["row_80_percent"] for seed in list_of_seeds]),
            "row_70_percent": np.std([reward_results_dict[epoch][seed]["row_70_percent"] for seed in list_of_seeds]),
            "row_60_percent": np.std([reward_results_dict[epoch][seed]["row_60_percent"] for seed in list_of_seeds]),
            "row_50_percent": np.std([reward_results_dict[epoch][seed]["row_50_percent"] for seed in list_of_seeds]),
            "row_40_percent": np.std([reward_results_dict[epoch][seed]["row_40_percent"] for seed in list_of_seeds]),
            "row_30_percent": np.std([reward_results_dict[epoch][seed]["row_30_percent"] for seed in list_of_seeds]),
            "row_20_percent": np.std([reward_results_dict[epoch][seed]["row_20_percent"] for seed in list_of_seeds]),
            "row_10_percent": np.std([reward_results_dict[epoch][seed]["row_10_percent"] for seed in list_of_seeds]),
        }

        std_params_dict[epoch] = {
            "row_max": np.std([param_results_dict[epoch][seed]["row_max"] for seed in list_of_seeds]),
            "row_99_percent": np.std([param_results_dict[epoch][seed]["row_99_percent"] for seed in list_of_seeds]),
            "row_90_percent": np.std([param_results_dict[epoch][seed]["row_90_percent"] for seed in list_of_seeds]),
            "row_80_percent": np.std([param_results_dict[epoch][seed]["row_80_percent"] for seed in list_of_seeds]),
            "row_70_percent": np.std([param_results_dict[epoch][seed]["row_70_percent"] for seed in list_of_seeds]),
            "row_60_percent": np.std([param_results_dict[epoch][seed]["row_60_percent"] for seed in list_of_seeds]),
            "row_50_percent": np.std([param_results_dict[epoch][seed]["row_50_percent"] for seed in list_of_seeds]),
            "row_40_percent": np.std([param_results_dict[epoch][seed]["row_40_percent"] for seed in list_of_seeds]),
            "row_30_percent": np.std([param_results_dict[epoch][seed]["row_30_percent"] for seed in list_of_seeds]),
            "row_20_percent": np.std([param_results_dict[epoch][seed]["row_20_percent"] for seed in list_of_seeds]),
            "row_10_percent": np.std([param_results_dict[epoch][seed]["row_10_percent"] for seed in list_of_seeds]),
        }

        min_params_dict[epoch] = {
            "row_max": np.min([param_results_dict[epoch][seed]["row_max"] for seed in list_of_seeds]),
            "row_99_percent": np.min([param_results_dict[epoch][seed]["row_99_percent"] for seed in list_of_seeds]),
            "row_90_percent": np.min([param_results_dict[epoch][seed]["row_90_percent"] for seed in list_of_seeds]),
            "row_80_percent": np.min([param_results_dict[epoch][seed]["row_80_percent"] for seed in list_of_seeds]),
            "row_70_percent": np.min([param_results_dict[epoch][seed]["row_70_percent"] for seed in list_of_seeds]),
            "row_60_percent": np.min([param_results_dict[epoch][seed]["row_60_percent"] for seed in list_of_seeds]),
            "row_50_percent": np.min([param_results_dict[epoch][seed]["row_50_percent"] for seed in list_of_seeds]),
            "row_40_percent": np.min([param_results_dict[epoch][seed]["row_40_percent"] for seed in list_of_seeds]),
            "row_30_percent": np.min([param_results_dict[epoch][seed]["row_30_percent"] for seed in list_of_seeds]),
            "row_20_percent": np.min([param_results_dict[epoch][seed]["row_20_percent"] for seed in list_of_seeds]),
            "row_10_percent": np.min([param_results_dict[epoch][seed]["row_10_percent"] for seed in list_of_seeds]),
        }

        max_params_dict[epoch] = {
            "row_max": np.max([param_results_dict[epoch][seed]["row_max"] for seed in list_of_seeds]),
            "row_99_percent": np.max([param_results_dict[epoch][seed]["row_99_percent"] for seed in list_of_seeds]),
            "row_90_percent": np.max([param_results_dict[epoch][seed]["row_90_percent"] for seed in list_of_seeds]),
            "row_80_percent": np.max([param_results_dict[epoch][seed]["row_80_percent"] for seed in list_of_seeds]),
            "row_70_percent": np.max([param_results_dict[epoch][seed]["row_70_percent"] for seed in list_of_seeds]),
            "row_60_percent": np.max([param_results_dict[epoch][seed]["row_60_percent"] for seed in list_of_seeds]),
            "row_50_percent": np.max([param_results_dict[epoch][seed]["row_50_percent"] for seed in list_of_seeds]),
            "row_40_percent": np.max([param_results_dict[epoch][seed]["row_40_percent"] for seed in list_of_seeds]),
            "row_30_percent": np.max([param_results_dict[epoch][seed]["row_30_percent"] for seed in list_of_seeds]),
            "row_20_percent": np.max([param_results_dict[epoch][seed]["row_20_percent"] for seed in list_of_seeds]),
            "row_10_percent": np.max([param_results_dict[epoch][seed]["row_10_percent"] for seed in list_of_seeds]),
        }        

    # subplot average and std rewards and params
    fig, axs = plt.subplots(1,2, figsize=(25, 15))
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_max"] for epoch in list_of_epochs], label="max")
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], label="99%")
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], label="90%")
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], label="80%")
    # axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], label="70%")
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], label="60%")
    # axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], label="50%")
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], label="40%")
    # axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], label="30%")
    axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], label="20%")
    # axs[0].plot(list_of_epochs, [average_rewards_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], label="10%")

    # fill in the area between the lines with the std
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_max"] - std_rewards_dict[epoch]["row_max"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_max"] + std_rewards_dict[epoch]["row_max"] for epoch in list_of_epochs], alpha=0.2)
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_99_percent"] - std_rewards_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_99_percent"] + std_rewards_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_90_percent"] - std_rewards_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_90_percent"] + std_rewards_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_80_percent"] - std_rewards_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_80_percent"] + std_rewards_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_70_percent"] - std_rewards_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_70_percent"] + std_rewards_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_60_percent"] - std_rewards_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_60_percent"] + std_rewards_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_50_percent"] - std_rewards_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_50_percent"] + std_rewards_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_40_percent"] - std_rewards_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_40_percent"] + std_rewards_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_30_percent"] - std_rewards_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_30_percent"] + std_rewards_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_20_percent"] - std_rewards_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_20_percent"] + std_rewards_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[0].fill_between(list_of_epochs, [average_rewards_dict[epoch]["row_10_percent"] - std_rewards_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], [average_rewards_dict[epoch]["row_10_percent"] + std_rewards_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], alpha=0.2)

    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("reward")
    axs[0].legend()

    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_max"] for epoch in list_of_epochs], label="max")
    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], label="99%")
    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], label="90%")
    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], label="80%")
    # axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], label="70%")
    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], label="60%")
    # axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], label="50%")
    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], label="40%")
    # axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], label="30%")
    axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], label="20%")
    # axs[1].plot(list_of_epochs, [average_params_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], label="10%")

    # # fill in the area between the lines with the std
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_max"] - std_params_dict[epoch]["row_max"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_max"] + std_params_dict[epoch]["row_max"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_99_percent"] - std_params_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_99_percent"] + std_params_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_90_percent"] - std_params_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_90_percent"] + std_params_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_80_percent"] - std_params_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_80_percent"] + std_params_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], alpha=0.2)
    # # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_70_percent"] - std_params_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_70_percent"] + std_params_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_60_percent"] - std_params_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_60_percent"] + std_params_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], alpha=0.2)
    # # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_50_percent"] - std_params_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_50_percent"] + std_params_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_40_percent"] - std_params_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_40_percent"] + std_params_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], alpha=0.2)
    # # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_30_percent"] - std_params_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_30_percent"] + std_params_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_20_percent"] - std_params_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_20_percent"] + std_params_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], alpha=0.2)
    # # axs[1].fill_between(list_of_epochs, [average_params_dict[epoch]["row_10_percent"] - std_params_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], [average_params_dict[epoch]["row_10_percent"] + std_params_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], alpha=0.2)

    # fill in the area between the lines with the std
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_max"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_max"] for epoch in list_of_epochs], alpha=0.2)
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_99_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_90_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_80_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_70_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_60_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_50_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_40_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_30_percent"] for epoch in list_of_epochs], alpha=0.2)
    axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_20_percent"] for epoch in list_of_epochs], alpha=0.2)
    # axs[1].fill_between(list_of_epochs, [min_params_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], [max_params_dict[epoch]["row_10_percent"] for epoch in list_of_epochs], alpha=0.2)


    # make y axis logarithmic
    axs[1].set_yscale("log")

    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("log param")
    axs[1].legend()

    os.makedirs(config.dest, exist_ok=True)
    fig.savefig(os.path.join(config.dest, "average_rewards_and_params.png"))
    
    
    pass

    
if __name__ == "__main__":
    config = get_config()
    main(config)