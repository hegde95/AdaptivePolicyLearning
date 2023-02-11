
import argparse
import os
import pandas as pd
import functools as ft
import matplotlib.pyplot as plt

import numpy as np
from plot_utils.sac_plotter import plot_model_dict






def get_config():
    parser = argparse.ArgumentParser(description='Plot archwise plot accross epochs')
    # parser.add_argument('--runs', type=str, help='Comma separated list of runs to plot')
    parser.add_argument('--run_dir', default = "runs_from_brain", type=str, help='Directory where the runs are stored')
    parser.add_argument('--dest', default="./reports/", type=str, help='Destination folder to save reports')
    # parser.add_argument('--prefix', default="", type=str, help='Prefix to add to the plot name')

    return parser.parse_args()


def main(config):
    list_of_runs = [os.path.join(*[config.run_dir, run]) for run in config.runs.split(',')]

    # list all model_dicts from these runs
    list_of_path_to_model_dicts = [os.path.join(*[run, 'model_dicts']) for run in list_of_runs]

    # list all model_dicts in each of these folders
    datas = []
    i_s = []
    for i,path_to_model_dicts in enumerate(list_of_path_to_model_dicts):
        list_of_model_dicts = [os.path.join(path_to_model_dicts, model_dict) for model_dict in os.listdir(path_to_model_dicts)]

        # sort this list
        list_of_model_dicts.sort(key = lambda x:int(x.split("dict")[2].split(".")[0]))

        # remove if epoch is greater than 3000000
        list_of_model_dicts = [model_dict for model_dict in list_of_model_dicts if int(model_dict.split("dict")[2].split(".")[0]) < 3000000]

        # get last model_dict
        last_model_dict = list_of_model_dicts[-1]

        # load model_dict
        data = pd.read_csv(last_model_dict)
        # data[f"architecture_{i}"] = data["architecture"]
        # data[f"reward_{i}"] = data["reward"]
        # data[f"params_{i}"] = data["params"]
        # data[f"capacity_{i}"] = data["capacity"]
        # rename columns
        data = data.rename(columns={"reward": f"reward_{i}", "params": f"params_{i}", "capacity": f"capacity_{i}"})
        datas.append(data)
        i_s.append(i)
    
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='architecture'), datas)

    # average over i_s into a new dataframe
    df_avg = pd.DataFrame()
    df_avg["reward"] = df_final[[f"reward_{i}" for i in i_s]].mean(axis=1)
    df_avg["params"] = df_final[[f"params_{i}" for i in i_s]].mean(axis=1)
    df_avg["capacity"] = df_final[[f"capacity_{i}" for i in i_s]].mean(axis=1)
    df_avg["architecture"] = df_final["architecture"]

    # remove negative rewards
    df_avg = df_avg[df_avg["reward"] > 0]

    path_to_report_folder = os.path.join(config.dest, config.prefix)
    os.makedirs(path_to_report_folder, exist_ok=True)
    env_name_dict = {
        "humanoid":"Humanoid-v2",
        "ant":"Ant-v2",
        "hc":"HalfCheetah-v2",
        "hopper":"Hopper-v2",
        "walker":"Walker2d-v2"        
    }
    inp_out_dict = {
        "humanoid":(376, 17),
        "ant":(111, 8),
        "hc":(17,6),
        "hopper":(11,3),
        "walker":(17,6)
    }    
    inp_dim = inp_out_dict[config.prefix][0]
    out_dim = inp_out_dict[config.prefix][1]
    env_name = env_name_dict[config.prefix]
    rvc_folder = os.path.join(path_to_report_folder, "rvc")
    rvp_folder = os.path.join(path_to_report_folder, "rvp")
    os.makedirs(rvc_folder, exist_ok=True)
    os.makedirs(rvp_folder, exist_ok=True)
    plot_model_dict(df_avg, env_name, 3000000, rvp_folder, rvc_folder)

    # create a normalized reward column
    df_avg['norm_reward'] = (df_avg['reward'] - df_avg['reward'].min()) / (df_avg['reward'].max() - df_avg['reward'].min())  

    # get max reward
    max_reward = df_avg.norm_reward.max()

    # get those with greater than 0.9 reward
    high_performing_archs = df_avg[df_avg.norm_reward >= 0.9*max_reward]

    # choose the one with least params
    smallest_param = min(high_performing_archs.params)
    tmp = high_performing_archs[high_performing_archs.params == smallest_param]

    # among these, choose the one with better reward
    tmp = tmp[tmp.norm_reward == tmp.norm_reward.max()]

    # save this as a csv
    tmp.to_csv(os.path.join(path_to_report_folder, "best_arch.csv"), index=False)

    # save avg df as csv
    df_avg.to_csv(os.path.join(path_to_report_folder, "avg_arch.csv"), index=False)

    task_wise_param_cutoff = {
        "humanoid": 4000, #3874,
        "ant": 4000, #3604,
        "hc": 2500, #1988,
        "hopper": 500, #202,
        "walker": 1000, #688
    }
    # just checking
    df_avg = df_avg[df_avg['params'] <= task_wise_param_cutoff[config.prefix]]

    list_of_layers = [4,8,16,32,64,128,256,512]
    largest_layer = 0
    for layer in list_of_layers:
        if layer * (inp_dim + 2) <= df_avg['capacity'].max():
            largest_layer = layer

    def check_if_arc_starts_with(layer, arch):
        arch = [int(layer) for layer in arch.strip("[]").split(", ")]
        if arch[0] == layer:
            return True
        else:
            return False
    
    # get all architectures that start with largest_layer
    pred_df_avg = df_avg[df_avg['architecture'].apply(lambda x: check_if_arc_starts_with(largest_layer, x))]

    # predict best model
    pred_df_avg = pred_df_avg[pred_df_avg['params'] == pred_df_avg['params'].max()]

    # actual best model
    act_df_avg = df_avg[df_avg['reward'] == df_avg['reward'].max()]

    print(f"for {config.prefix}, params cutoff is {task_wise_param_cutoff[config.prefix]}")

    print(f"Predicted best architecture: {pred_df_avg['architecture'].values[0]} with reward: {pred_df_avg['reward'].values[0]} and params: {pred_df_avg['params'].values[0]}")
    print(f"Actual best architecture: {act_df_avg['architecture'].values[0]} with reward: {act_df_avg['reward'].values[0]} and params: {act_df_avg['params'].values[0]}")

    # sort by reward
    df_avg = df_avg.sort_values(by=['reward'], ascending=False)
    # save cutoff df as csv
    df_avg.to_csv(os.path.join(path_to_report_folder, "cutoff_arch.csv"), index=False)

    # plot reward vs capacity
    plt.figure()
    plt.scatter(np.log(df_avg['capacity']), df_avg['reward'], c=np.log(df_avg['params']), s=10, cmap = 'coolwarm')
    plt.xlabel("ln(Capacity)")
    plt.ylabel("Reward")
    plt.title(f"Reward vs Capacity for {env_name}, given a constraint of {task_wise_param_cutoff[config.prefix]} parameters")
    plt.savefig(os.path.join(path_to_report_folder, "const_reward_vs_capacity.png"))
    print("\n\n")
    pass

if __name__ == "__main__":
    config = get_config()
    os.makedirs(config.dest, exist_ok=True)
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
