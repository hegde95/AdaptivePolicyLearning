# AdaptivePolicyLearning



To run from config file:

    python main.py --cuda True --seed 123 --base_dir runs --config ant

Suggested override options for config runs:

    --seed, "set seed for random number generator"
    --cuda, "use cuda"
    --cuda_device, "set cuda device"
    --debug, "run in debug mode" 
    --hyper, "use hyper policy, default with config is True"
    --parallel, "use an ensemble of parallel policies, default with config is False"
    --wandb, "use wandb, default with config is True" 
    --wandb-tag, "set wandb tag, default with config is bm8"
    --base_dir, "set base directory for runs, default with config is runs2"

Baseline run
    python main.py --hyper False --base_dir runs_size_check --config ant --arc 16,32,32,4 --wandb-tag bsc --seed 111

    python main.py --hyper False --base_dir runs_size_check --config hopper --arc 8,4,4,4 --wandb-tag bsc --seed 111

To resume training from a previous run:

    python main.py --cuda True --load_run 2022-08-19_16-24-14_DEBUG_Ant-v2_Gaussian__hyper_123
    
To train on a manipulation task, we use ddpg with hindsight experience replay (HER) and a goal-conditioned policy. To run HER from config file:

    python main_ddpg.py --cuda True --seed 123 --base_dir runs --config fetchpickplace


To plot results for a  sac run:

    python plot_utils/sac_plotter.py --eval_every 4 --run_dir runs2 --run_name 2022-08-19_16-24-14_DEBUG_Ant-v2_Gaussian__hyper_123

To create a percentage plot accross runs:

    python plot_utils/bar_plotter.py --runs 2022-07-05_01-18-41_SAC_HalfCheetah-v2_Gaussian__hyper_111,2022-07-05_01-18-49_SAC_HalfCheetah-v2_Gaussian__hyper_222,2022-07-05_01-18-59_SAC_HalfCheetah-v2_Gaussian__hyper_333,2022-07-05_01-19-04_SAC_HalfCheetah-v2_Gaussian__hyper_444,2022-07-05_01-19-11_SAC_HalfCheetah-v2_Gaussian__hyper_555,2022-07-05_01-19-25_SAC_HalfCheetah-v2_Gaussian__hyper_666


