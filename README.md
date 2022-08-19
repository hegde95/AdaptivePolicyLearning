# AdaptivePolicyLearning



To run from config file:

    python main.py --cuda True --seed 123 --base_dir runs --config ant


To resume training from a previous run:

    python main.py --cuda True --load_run 2022-08-19_16-24-14_DEBUG_Ant-v2_Gaussian__hyper_123
    


To plot results for a run:

    python plot_utils/plotter.py --eval_every 4 --run_dir runs2 --run_name 2022-08-19_16-24-14_DEBUG_Ant-v2_Gaussian__hyper_123

To create a percentage plot accross runs:

    python plot_utils/bar_plotter.py --runs 2022-07-05_01-18-41_SAC_HalfCheetah-v2_Gaussian__hyper_111,2022-07-05_01-18-49_SAC_HalfCheetah-v2_Gaussian__hyper_222,2022-07-05_01-18-59_SAC_HalfCheetah-v2_Gaussian__hyper_333,2022-07-05_01-19-04_SAC_HalfCheetah-v2_Gaussian__hyper_444,2022-07-05_01-19-11_SAC_HalfCheetah-v2_Gaussian__hyper_555,2022-07-05_01-19-25_SAC_HalfCheetah-v2_Gaussian__hyper_666