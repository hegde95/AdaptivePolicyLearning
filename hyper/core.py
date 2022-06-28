import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Normal
from torch.distributions.normal import Normal

from hyper.ghn_modules import MLP_GHN, MlpNetwork
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from itertools import product as cartesian_product
from hyper.plotter import get_capacity
import random

class hyperActor(nn.Module):

    def __init__(self, 
                act_dim, 
                obs_dim, 
                act_limit, 
                allowable_layers, 
                search = False, 
                conditional = False, 
                meta_batch_size = 1,
                gumbel_tau = 1.0,
                ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.is_search = search
        self.conditional = conditional
        self.meta_batch_size = meta_batch_size

        list_of_allowable_layers = list(allowable_layers)
        list_of_allowable_layers.append(0)
        list_of_allowable_layers.sort()
        self.list_of_allowable_layers = torch.Tensor(list_of_allowable_layers).to('cuda')

        if self.is_search:
            self.tau = gumbel_tau
            # TODO : change the length of these parameters to match list of allowable layers
            self.base_inp_to_layer1_dist = nn.Parameter(torch.ones(len(self.list_of_allowable_layers) - 1).to('cuda'), requires_grad=True)
            self.base_inp_to_layer2_dist = nn.Parameter(torch.ones(len(self.list_of_allowable_layers)).to('cuda'), requires_grad=True)
            self.base_inp_to_layer3_dist = nn.Parameter(torch.ones(len(self.list_of_allowable_layers)).to('cuda'), requires_grad=True)    
            self.base_inp_to_layer4_dist = nn.Parameter(torch.ones(len(self.list_of_allowable_layers)).to('cuda'), requires_grad=True)    

            if self.conditional:
                self.conditional_layer1_distribution = nn.Sequential(nn.Linear(8, 8), nn.ReLU(),nn.Linear(8, 8)).to('cuda')
                self.conditional_layer2_distribution = nn.Sequential(nn.Linear(10, 9), nn.ReLU(),nn.Linear(9, 9)).to('cuda')
                self.conditional_layer3_distribution = nn.Sequential(nn.Linear(10, 9), nn.ReLU(),nn.Linear(9, 9)).to('cuda')
                self.conditional_layer4_distribution = nn.Sequential(nn.Linear(10, 9), nn.ReLU(),nn.Linear(9, 9)).to('cuda')


        config = {}
        config['max_shape'] = (512, 512, 1, 1)
        config['num_classes'] = 4 * act_dim
        config['num_observations'] = obs_dim
        config['weight_norm'] = True
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = 16
        self.ghn_config = config


        self.ghn = MLP_GHN(**config,
                    debug_level=0).to('cuda')  

        self.optimizer = torch.optim.Adam(self.ghn.parameters(), 1e-3, weight_decay=1e-5)
        if self.is_search:
            if self.conditional:
                self.search_optimizer = torch.optim.Adam([
                    {
                        'params':[
                            self.base_inp_to_layer1_dist, 
                            self.base_inp_to_layer2_dist, 
                            self.base_inp_to_layer3_dist, 
                            self.base_inp_to_layer4_dist, 
                        ], 
                        'lr':1e-3
                    },
                    {
                        'params':self.conditional_layer1_distribution.parameters(),
                        'lr':1e-3
                    },
                    {
                        'params':self.conditional_layer2_distribution.parameters(),
                        'lr':1e-3
                    },
                    {
                        'params':self.conditional_layer3_distribution.parameters(),
                        'lr':1e-3
                    },
                    {
                        'params':self.conditional_layer4_distribution.parameters(),
                        'lr':1e-3
                    }
                    ])
            else:
                self.search_optimizer = torch.optim.Adam([
                    {
                        'params':[
                            self.base_inp_to_layer1_dist,
                            self.base_inp_to_layer2_dist,
                            self.base_inp_to_layer3_dist,
                            self.base_inp_to_layer4_dist,
                        ],
                        'lr':8e-3
                    }
                    ])
        self.scheduler = MultiStepLR(self.optimizer, milestones='200,250', gamma=0.1)
        self.change_graph()

        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.act_limit = act_limit



    def re_query_biased_weights(self):
        self.current_model = []
        shape_inds = []
        param_counts = []
        for i in range(self.meta_batch_size):
            fc_layers = []
            shape_ind = [torch.tensor(0).to('cuda')]
            param_count = torch.tensor(0.0).to('cuda')
            # layer 1
            if self.conditional:
                self.layer_1_sample = self.conditional_layer1_distribution(self.base_inp_to_layer1_dist)
                self.layer_1_sample = torch.matmul(F.gumbel_softmax(self.layer_1_sample,hard = True, tau=self.tau), self.list_of_allowable_layers[1:])
            else:
                self.layer_1_sample = torch.matmul(F.gumbel_softmax(self.base_inp_to_layer1_dist,hard = True, tau=self.tau), self.list_of_allowable_layers[1:])
            self.layer_1_actual = int(self.layer_1_sample.item())
            fc_layers.append(self.layer_1_actual)
            shape_ind.append(self.layer_1_sample) 
            shape_ind.append(self.layer_1_sample) 
            param_count += (torch.tensor((self.obs_dim)).to('cuda') + 1) * self.layer_1_sample

            # layer 2
            if self.conditional:
                self.layer_2_sample = self.conditional_layer2_distribution(torch.cat([self.base_inp_to_layer2_dist, self.layer_1_sample.detach().view(-1)]))
                self.layer_2_sample = torch.matmul(F.gumbel_softmax(self.layer_2_sample,hard = True, tau=self.tau), self.list_of_allowable_layers)
            else:
                self.layer_2_sample = torch.matmul(F.gumbel_softmax(self.base_inp_to_layer2_dist,hard = True, tau=self.tau), self.list_of_allowable_layers)
            self.layer_2_actual = int(self.layer_2_sample.item())
            if self.layer_2_sample.item() > 0:
                fc_layers.append(self.layer_2_actual)
                shape_ind.append(self.layer_2_sample)
                shape_ind.append(self.layer_2_sample)
                param_count += ((self.layer_1_sample + 1) * self.layer_2_sample)

                # layer 3
                if self.conditional:
                    self.layer_3_sample = self.conditional_layer3_distribution(torch.cat([self.base_inp_to_layer3_dist, self.layer_2_sample.detach().view(-1)]))
                    self.layer_3_sample = torch.matmul(F.gumbel_softmax(self.layer_3_sample,hard = True, tau=self.tau), self.list_of_allowable_layers)
                else:
                    self.layer_3_sample = torch.matmul(F.gumbel_softmax(self.base_inp_to_layer3_dist,hard = True, tau=self.tau), self.list_of_allowable_layers)
                self.layer_3_actual = int(self.layer_3_sample.item())
                if self.layer_3_sample.item() > 0:
                    fc_layers.append(self.layer_3_actual)
                    shape_ind.append(self.layer_3_sample)
                    shape_ind.append(self.layer_3_sample)
                    param_count += ((self.layer_2_sample + 1) * self.layer_3_sample)

                    # layer 4
                    if self.conditional:
                        self.layer_4_sample = self.conditional_layer4_distribution(torch.cat([self.base_inp_to_layer4_dist, self.layer_3_sample.detach().view(-1)]))
                        self.layer_4_sample = torch.matmul(F.gumbel_softmax(self.layer_4_sample,hard = True, tau=self.tau), self.list_of_allowable_layers)
                    else:
                        self.layer_4_sample = torch.matmul(F.gumbel_softmax(self.base_inp_to_layer4_dist,hard = True, tau=self.tau), self.list_of_allowable_layers)
                    self.layer_4_actual = int(self.layer_4_sample.item())
                    if self.layer_4_sample.item() > 0:
                        fc_layers.append(self.layer_4_actual)
                        shape_ind.append(self.layer_4_sample)
                        shape_ind.append(self.layer_4_sample)
                        param_count += ((self.layer_3_sample + 1) * self.layer_4_sample)
                        param_count += ((self.layer_4_sample + 1) * torch.tensor((self.act_dim * 2)).to('cuda'))

                else:
                    self.layer_4_actual = 0
                    param_count += ((self.layer_2_sample + 1) * torch.tensor((self.act_dim * 2)).to('cuda'))

            else:
                self.layer_3_actual = 0
                self.layer_4_actual = 0
                param_count += ((self.layer_1_sample + 1) * torch.tensor((self.act_dim * 2)).to('cuda'))

                
            shape_ind.append(torch.tensor((self.act_dim * 2)).to('cuda'))
            shape_ind.append(torch.tensor((self.act_dim * 2)).to('cuda'))
            net = MlpNetwork(fc_layers=fc_layers, inp_dim = self.obs_dim, out_dim = 2 * self.act_dim)
            self.current_model.append(net)
            shape_ind = torch.stack(shape_ind).view(-1,1)
            shape_inds.append(shape_ind)
            param_counts.append(param_count)

        shape_inds = torch.cat(shape_inds) 
        self.current_capacity = get_capacity(fc_layers, self.obs_dim, self.act_dim)
        self.current_number_of_params = sum(p.numel() for p in self.current_model[0].parameters())
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = shape_inds)
        self.param_counts = torch.stack(param_counts)
    
    def re_query_uniform_weights(self, repeat_sample = False):
        self.current_model = []
        shape_inds = []
        for i in range(self.meta_batch_size):
            self.create_random_net(repeat_sample)
            net = MlpNetwork(**self.net_args)
            self.current_model.append(net)
            self.layer_1_actual = 0
            self.layer_2_actual = 0
            self.layer_3_actual = 0
            self.layer_4_actual = 0

            shape_ind = [[0]]
            for i, layer in enumerate(self.net_args['fc_layers']):
                if i == 0:
                    self.layer_1_actual = layer
                if i == 1:
                    self.layer_2_actual = layer
                if i == 2:
                    self.layer_3_actual = layer
                if i == 3:
                    self.layer_4_actual = layer
                shape_ind.append([layer])
                shape_ind.append([layer])
            shape_ind.append([self.net_args['out_dim']])
            shape_ind.append([self.net_args['out_dim']])     
            shape_ind = torch.Tensor(shape_ind).to('cuda')     
            shape_inds.append(shape_ind)
        shape_inds = torch.cat(shape_inds)    
        self.current_capacity = get_capacity(self.net_args['fc_layers'], self.obs_dim, self.act_dim)
        self.current_number_of_params = sum(p.numel() for p in self.current_model[0].parameters())
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = shape_inds)

    def change_graph(self, biased_sample = False, repeat_sample = False):
        if biased_sample:
            self.re_query_biased_weights()
        else:
            self.re_query_uniform_weights(repeat_sample)



    def create_random_net(self, repeat_sample = False):
        if not repeat_sample:
            num_layer = np.random.choice([1,2,3,4])
            self.fc_layers = list(np.random.choice(self.list_of_allowable_layers[1:].cpu().numpy().astype(int),num_layer))
            self.net_args = {
                'fc_layers':self.fc_layers,
                'inp_dim':self.obs_dim,
                'out_dim':2 * self.act_dim
            }

    def get_current_layer_dist(self):
        if self.is_search:
            if self.conditional:
                # layer_1_logits = self.conditional_layer1_distribution(self.base_inp_to_layer1_dist)
                # layer_1_probs = F.softmax(layer_1_logits)

                # layer_2_logits = self.conditional_layer2_distribution(torch.cat([self.base_inp_to_layer2_dist, layer_1_sample.detach().view(-1)]))
                # layer_2_probs = F.softmax(layer_2_logits)
                raise NotImplementedError
            else:
                with torch.no_grad():
                    layer_1_logits = self.base_inp_to_layer1_dist
                    layer_1_probs = F.softmax(layer_1_logits)
                    layer_2_logits = self.base_inp_to_layer2_dist
                    layer_2_probs = F.softmax(layer_2_logits)
                    layer_3_logits = self.base_inp_to_layer3_dist
                    layer_3_probs = F.softmax(layer_3_logits)
                    layer_4_logits = self.base_inp_to_layer4_dist
                    layer_4_probs = F.softmax(layer_4_logits)

                return layer_1_probs, layer_2_probs, layer_3_probs, layer_4_probs


    def forward(self, state):
        # x = torch.stack([model(state) for model in self.current_model]).mean(dim=0)
        batch_per_net = int(state.shape[0]//self.meta_batch_size)
        x = torch.cat([self.current_model[i](state[i*batch_per_net:(i+1)*batch_per_net]) for i in range(self.meta_batch_size)])

        if len(x.shape) == 1:    
            mu = x[:x.shape[-1]//2]
            log_std = x[x.shape[-1]//2:]
        else:
            mu = x[:,:x.shape[-1]//2]
            log_std = x[:,x.shape[-1]//2:]


        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)        
        return mu, log_std
    

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob


    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)
    

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


    def get_logprob(self,obs, actions, epsilon=1e-6):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(1, keepdim=True)
        return log_prob