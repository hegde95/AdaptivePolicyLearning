from hyper.core import hyperActor
from hyper.ghn_modules import MLP_GHN, MlpNetwork
import numpy as np
import torch

import gym


if __name__ == "__main__":
    env_name = "HalfCheetah-v2"
    ckp_file = "checkpoints/HyperAgent"

    
    env = gym.make(env_name)
    env.seed(1234)

    num_inputs = env.observation_space.shape[0]
    env_act_dim = env.action_space.shape[0]
    action_space = env.action_space
    device = "cpu"

    actor = hyperActor(action_space.shape[0], num_inputs, action_space.high[0], np.arange(4,512 + 1))
    state_dict = torch.load(ckp_file, map_location='cpu')
    actor.load_state_dict(state_dict['ghn'])
    ghn = actor.ghn
    ghn = ghn.to('cpu')
    ghn.default_edges = ghn.default_edges.to('cpu')
    ghn.default_node_feat = ghn.default_node_feat.to('cpu')
    
    inp_dim = num_inputs
    out_dim = 2 * env_act_dim
    net_args = {
        'fc_layers':[],
        'inp_dim':inp_dim,
        'out_dim':out_dim,
    }  

    # num_layer = 3 #np.random.choice([1,2,3,4])
    arc = [512, 512, 32] #list(np.random.choice(np.arange(4,512),num_layer))
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

    params = sum(p.numel() for p in model.parameters())

    print(f"Number of params in model is {np.log10(params)}")

    for i in range(1000):
        done = False
        state = env.reset()
        ep_reward = 0
        while not done:
            state_t = torch.FloatTensor(state)
            action_t = model(state_t)
            action_t = action_t[:action_t.shape[-1]//2]
            env.render()

            action = action_t.detach().numpy()

            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            state = next_state
        
        print(f"Ep {i} reward: {ep_reward}")
        