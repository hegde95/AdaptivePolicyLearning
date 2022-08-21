from copy import copy, deepcopy
import os, json
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', ):
        return True
    elif isinstance(v, str) and v.lower() in ('false', ):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')



def get_args(parser):
    # general args
    parser.add_argument('--env-name',
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--seed', type=int,  metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--eval', type=str2bool, 
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--num_steps', type=int,  metavar='N',
                        help='maximum number of steps (default: 3000000)')
    parser.add_argument('--cuda', type = str2bool,
                        help='run on CUDA (default: False)')
    parser.add_argument('--cuda_device', type=int, 
                    help="sets the cuda device to run experiments on")
    parser.add_argument('--debug', type = str2bool,
                        help='Will run in debug. (default: False')  

    # sac args
    parser.add_argument('--policy',
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=str2bool, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--batch_size', type=int, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--taper', type = str2bool,
                        help='Taper the model shape (default: False)')
    parser.add_argument('--start_steps', type=int, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, metavar='N',
                        help='size of replay buffer (default: 1000000)')

    # hyper args
    parser.add_argument('--meta_batch_size', type=int, metavar='N',
                    help='hidden size (default: 8)')    
    parser.add_argument('--updates_per_step', type=int, metavar='N',
                        help='model updates per simulator step (default: 8)')
    parser.add_argument('--hyper', type = str2bool,
                        help='run with a hyper network (default: False)') 
    parser.add_argument('--parallel', type = str2bool,
                        help='run with an ensemble of networks (default: False)')
    parser.add_argument('--condition_q', type = str2bool,
                        help='condition the q network with the architecture (default: False)')   
    parser.add_argument('--steps_per_arc', type=int, metavar='N',
                        help='steps to run between architecture samples (default: 50)')
    parser.add_argument('--search', type = str2bool,
                        help = 'search for the best architecture (default: False)')

    # logging args 
    parser.add_argument('--wandb', type = str2bool,
                        help='Log to wandb. (default: False')  
    parser.add_argument('--wandb-tag', type=str,
                        help='Use a custom tag for wandb. (default: "")')                        
    parser.add_argument('--save_model', type = str2bool,
                    help="save the model after each episode")
    parser.add_argument('--load_run', type=str,
                        help='Load a run from latest checkpoint')
    parser.add_argument('--base_dir', type=str,
                        help='Base directory for the experiment (default: runs)')

    # dm control args
    parser.add_argument('--dm_control', type = str2bool,
                        help='run with dm control (default: False)')
    parser.add_argument('--domain', type=str,
                        help='domain to run dm control on (default: quadruped)')
    parser.add_argument('--task', type=str,
                        help='task to run dm control on (default: fetch)')


    parser.add_argument('--config', type=str, default="",
                        help = "Name of the config file to load, stored in the configs folder")

    return parser

def set_arg_defaults(config):
    config.env_name = config.env_name if config.env_name else "HalfCheetah-v2"
    config.seed = config.seed if config.seed else 123456
    config.eval = config.eval if config.eval else True
    config.num_steps = config.num_steps if config.num_steps else 3000000
    config.cuda = config.cuda if config.cuda else False
    config.cuda_device = config.cuda_device if config.cuda_device else 0
    config.debug = config.debug if config.debug else False
    config.policy = config.policy if config.policy else "Gaussian"
    config.gamma = config.gamma if config.gamma else 0.99
    config.tau = config.tau if config.tau else 0.005
    config.lr = config.lr if config.lr else 0.0003
    config.alpha = config.alpha if config.alpha else 0.2
    config.automatic_entropy_tuning = config.automatic_entropy_tuning if config.automatic_entropy_tuning else False
    config.batch_size = config.batch_size if config.batch_size else 256
    config.hidden_size = config.hidden_size if config.hidden_size else 256
    config.taper = config.taper if config.taper else False
    config.start_steps = config.start_steps if config.start_steps else 10000
    config.target_update_interval = config.target_update_interval if config.target_update_interval else 1
    config.replay_size = config.replay_size if config.replay_size else 1000000
    config.meta_batch_size = config.meta_batch_size if config.meta_batch_size else 8
    config.updates_per_step = config.updates_per_step if config.updates_per_step else 8
    config.hyper = config.hyper if config.hyper else False
    config.parallel = config.parallel if config.parallel else False
    config.condition_q = config.condition_q if config.condition_q else False
    config.steps_per_arc = config.steps_per_arc if config.steps_per_arc else 50
    config.search = config.search if config.search else False
    config.wandb = config.wandb if config.wandb else False
    config.wandb_tag = config.wandb_tag if config.wandb_tag else ""
    config.save_model = config.save_model if config.save_model else False
    config.load_run = config.load_run if config.load_run else None
    config.base_dir = config.base_dir if config.base_dir else "runs"
    config.dm_control = config.dm_control if config.dm_control else False
    config.domain = config.domain if config.domain else "quadruped"
    config.task = config.task if config.task else "fetch"
    
    return config

    




def override_config(args):
    orig_args = deepcopy(args)
    
    if args.config != "":
        # set config.json as default config
        config_path = os.path.join(os.getcwd(), "configs", args.config + ".json")
        with open(config_path) as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)

        # override config with command line args
        for key, value in vars(orig_args).items():
            if value is not None:
                setattr(args, key, value)
    else:
        # set default configs
        args.config = "default"
        args = set_arg_defaults(args)

        # override config with command line args
        for key, value in vars(orig_args).items():
            if value is not None:
                setattr(args, key, value)

    return args
