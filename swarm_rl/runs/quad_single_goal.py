from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [123]),
])

SMALL_MODEL_CLI = QUAD_BASELINE_CLI + (
    ' --train_for_env_steps=10000000000 --quads_mode=dynamic_same_goal --hidden_size=16 --neighbor_obs_type=none --quads_local_obs=-1 '
    '--quads_num_agents=1 --replay_buffer_sample_prob=0.0 --anneal_collision_steps=0 --save_milestones_sec=10000 '
    '--quads_neighbor_encoder_type=no_encoder --with_wandb=False --wandb_user=multi-drone'
)

_experiment = Experiment(
    'baseline',
    SMALL_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quad_single_baseline', experiments=[_experiment])

# python -m sample_factory.runner.run --run=swarm_rl.runs.quad_single_goal --runner=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1

# for brain
# sbatch launch/drone.sh