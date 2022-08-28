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
from itertools import product

class hyperActor(nn.Module):

    def __init__(self, 
                act_dim, 
                obs_dim, 
                act_limit, 
                allowable_layers, 
                search = False, 
                conditional = True, 
                meta_batch_size = 1,
                gumbel_tau = 1.0,
                device = "cpu"
                ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.is_search = search
        self.conditional = conditional
        self.meta_batch_size = meta_batch_size
        self.device = device


        list_of_allowable_layers = list(allowable_layers)

        self.list_of_arcs = []
        for k in range(1,5):
            # dum = list(cartesian_product(list_of_allowable_layers, repeat=k))
            # if k == 1:
            #     self.list_of_arcs.extend(dum)
            # elif k == 2:
            #     self.list_of_arcs.extend(list(filter(lambda x:(x[1]<=x[0]), dum)))
            # elif k == 3:
            #     self.list_of_arcs.extend(list(filter(lambda x:((x[1]<=x[0]) and (x[2]<=x[1])), dum)))
            # elif k == 4:
            #     self.list_of_arcs.extend(list(filter(lambda x:((x[1]<=x[0]) and (x[2]<=x[1]) and (x[3]<=x[2])), dum)))
            self.list_of_arcs.extend(list(product(list_of_allowable_layers, repeat = k)))
        self.list_of_arcs.sort(key = lambda x:self.get_params(x))

        # self.list_of_arcs_tensors = [torch.Tensor(arc).to(self.device) for arc in self.list_of_arcs]
        self.list_of_shape_inds = []
        for arc in self.list_of_arcs:
            shape_ind = [torch.tensor(0).type(torch.FloatTensor).to(self.device)]
            for layer in arc:
                shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
                shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim * 2).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim * 2).type(torch.FloatTensor).to(self.device))
            shape_ind = torch.stack(shape_ind).view(-1,1)
            self.list_of_shape_inds.append(shape_ind) 

        self.list_of_shape_inds_lenths = [x.squeeze().numel() for x in self.list_of_shape_inds]
        max_len = max(self.list_of_shape_inds_lenths)
        # pad -1 to the end of each shape_ind
        for i in range(len(self.list_of_shape_inds)):
            num_pad = (max_len - self.list_of_shape_inds[i].shape[0])
            self.list_of_shape_inds[i] = torch.cat([self.list_of_shape_inds[i], torch.tensor(-1).to(self.device).repeat(num_pad,1)], 0)
        self.list_of_shape_inds = torch.stack(self.list_of_shape_inds)
        self.list_of_shape_inds = self.list_of_shape_inds.reshape(len(self.list_of_shape_inds),max_len)
        self.list_of_arc_indices = np.arange(len(self.list_of_arcs))
        # shuffle the list of arcs indices
        np.random.shuffle(self.list_of_arc_indices)
        self.current_model_indices = np.arange(self.meta_batch_size)
        # self.list_of_shape_inds = self.list_of_shape_inds.permute(1,2,0)
        
        if self.is_search:

            list_of_allowable_layers.append(0)
            list_of_allowable_layers.sort()
            self.list_of_allowable_layers = torch.Tensor(list_of_allowable_layers).to(self.device)

            self.tau = gumbel_tau
            # self.conditional_layer1_distribution = nn.Sequential(nn.Linear(obs_dim, len(self.list_of_allowable_layers) - 1), nn.ReLU()).to(self.device)
            # self.conditional_layer2_distribution = nn.Sequential(nn.Linear(1 + obs_dim, len(self.list_of_allowable_layers)), nn.ReLU()).to(self.device)
            # self.conditional_layer3_distribution = nn.Sequential(nn.Linear(1 + obs_dim, len(self.list_of_allowable_layers)), nn.ReLU()).to(self.device)
            # self.conditional_layer4_distribution = nn.Sequential(nn.Linear(1 + obs_dim, len(self.list_of_allowable_layers)), nn.ReLU()).to(self.device)

            self.conditional_arc_dist = nn.Sequential(nn.Linear(obs_dim, len(self.list_of_shape_inds)//2), nn.ReLU(), nn.Linear(len(self.list_of_shape_inds)//2, len(self.list_of_shape_inds)), nn.Sigmoid()).to(self.device)


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
                    debug_level=0, device=self.device).to(self.device)  

        self.optimizer = torch.optim.Adam(self.ghn.parameters(), 1e-3, weight_decay=1e-5)
        if self.is_search:
            self.search_optimizer = torch.optim.Adam([
                # {
                #     'params':self.conditional_layer1_distribution.parameters(),
                #     'lr':1e-3
                # },
                # {
                #     'params':self.conditional_layer2_distribution.parameters(),
                #     'lr':1e-3
                # },
                # {
                #     'params':self.conditional_layer3_distribution.parameters(),
                #     'lr':1e-3
                # },
                # {
                #     'params':self.conditional_layer4_distribution.parameters(),
                #     'lr':1e-3
                # },


                {
                    'params':self.conditional_arc_dist.parameters(),
                    'lr':1e-3
                }
                ])

        self.scheduler = MultiStepLR(self.optimizer, milestones=[200,250], gamma=0.1)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[100,150], gamma=0.1)
        # self.change_graph()

        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.act_limit = act_limit

    def re_query_biased_weights(self, state, eval = False):
        logits = self.conditional_arc_dist(state)
        if eval:
            self.sampled_indices = torch.argmax(F.softmax(logits, dim = -1), 1)
            self.sampled_shape_inds = torch.cat([self.list_of_shape_inds[index][:self.list_of_shape_inds_lenths[index]] for index in self.sampled_indices]).view(-1,1)
        else:
            self.sampled_one_hot = F.gumbel_softmax(logits,hard = True, tau=self.tau)
            self.sampled_indices = torch.argmax(self.sampled_one_hot, 1)
            self.sampled_shape_inds = torch.matmul(self.sampled_one_hot, self.list_of_shape_inds)
            self.sampled_shape_inds = torch.cat([self.sampled_shape_inds[i][:self.list_of_shape_inds_lenths[self.sampled_indices[i]]] for i in range(len(self.sampled_indices))]).view(-1,1)
        
        self.current_model = [MlpNetwork(fc_layers=self.list_of_arcs[index], inp_dim = self.obs_dim, out_dim = 2 * self.act_dim) for index in self.sampled_indices]
        self.param_counts = [self.get_params(self.list_of_arcs[index]) for index in self.sampled_indices]
        self.capacities = [get_capacity(self.list_of_arcs[index], self.obs_dim, self.act_dim) for index in self.sampled_indices]
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)

    def set_graph(self, graph):
        # graph has to be list of list of layer, eg [[32,16,8],[4,128,4]]
        size = len(graph)
        shape_inds = []
        self.current_model = []
        self.param_counts = []
        self.capacities = []
        for i in range(size):
            shape_ind = [torch.tensor(0).to(self.device)]
            for j in range(len(graph[i])):
                shape_ind.append(torch.tensor(graph[i][j]).type(torch.FloatTensor).to(self.device))
                shape_ind.append(torch.tensor(graph[i][j]).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor((self.act_dim * 2)).to(self.device))
            shape_ind.append(torch.tensor((self.act_dim * 2)).to(self.device))
            shape_ind = torch.stack(shape_ind).view(-1,1)
            shape_inds.append(shape_ind)
            self.current_model.append(MlpNetwork(fc_layers=graph[i], inp_dim = self.obs_dim, out_dim = 2 * self.act_dim))
            self.param_counts.append(self.get_params(graph[i]))
            self.capacities.append(get_capacity(graph[i], self.obs_dim, self.act_dim))
        self.sampled_shape_inds = torch.cat(shape_inds)
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)

    # def re_query_biased_weights2(self, state, eval = False):
    #     self.layer_1_sample = self.conditional_layer1_distribution(state)
    #     if eval:
    #         self.layer_1_sample = self.list_of_allowable_layers[1:][torch.argmax(F.softmax(self.layer_1_sample), 1)]
    #     else:    
    #         self.layer_1_sample = torch.matmul(F.gumbel_softmax(self.layer_1_sample,hard = True, tau=self.tau), self.list_of_allowable_layers[1:])

    #     self.layer_2_sample = self.conditional_layer2_distribution(torch.cat((self.layer_1_sample.view(-1,1).detach(), state), dim =1))
    #     if eval:
    #         self.layer_2_sample = self.list_of_allowable_layers[torch.argmax(F.softmax(self.layer_2_sample), 1)]
    #     else:
    #         self.layer_2_sample = torch.matmul(F.gumbel_softmax(self.layer_2_sample,hard = True, tau=self.tau), self.list_of_allowable_layers)

    #     self.layer_3_sample = self.conditional_layer3_distribution(torch.cat((self.layer_2_sample.view(-1,1).detach(), state), dim =1))
    #     if eval:
    #         self.layer_3_sample = self.list_of_allowable_layers[torch.argmax(F.softmax(self.layer_3_sample), 1)]
    #     else:
    #         self.layer_3_sample = torch.matmul(F.gumbel_softmax(self.layer_3_sample,hard = True, tau=self.tau), self.list_of_allowable_layers)

    #     self.layer_4_sample = self.conditional_layer4_distribution(torch.cat((self.layer_3_sample.view(-1,1).detach(), state), dim =1))
    #     if eval:
    #         self.layer_4_sample = self.list_of_allowable_layers[torch.argmax(F.softmax(self.layer_4_sample), 1)]
    #     else:
    #         self.layer_4_sample = torch.matmul(F.gumbel_softmax(self.layer_4_sample,hard = True, tau=self.tau), self.list_of_allowable_layers)

    #     self.current_model = []
    #     shape_inds = []
    #     self.param_counts = []
    #     self.capacities = []
    #     for i in range(state.shape[0]):
    #         fc_layers = []
    #         shape_ind = [torch.tensor(0).to(self.device)]
    #         shape_ind.append(self.layer_1_sample[i])
    #         shape_ind.append(self.layer_1_sample[i])
    #         fc_layers.append(int(self.layer_1_sample[i].item()))
    #         if self.layer_2_sample[i] > 0:
    #             shape_ind.append(self.layer_2_sample[i])
    #             shape_ind.append(self.layer_2_sample[i])
    #             fc_layers.append(int(self.layer_2_sample[i].item()))
    #             if self.layer_3_sample[i] > 0:
    #                 shape_ind.append(self.layer_3_sample[i])
    #                 shape_ind.append(self.layer_3_sample[i])
    #                 fc_layers.append(int(self.layer_3_sample[i].item()))
    #                 if self.layer_4_sample[i] > 0:
    #                     shape_ind.append(self.layer_4_sample[i])
    #                     shape_ind.append(self.layer_4_sample[i])
    #                     fc_layers.append(int(self.layer_4_sample[i].item()))
    #         shape_ind.append(torch.tensor((self.act_dim * 2)).to(self.device))
    #         shape_ind.append(torch.tensor((self.act_dim * 2)).to(self.device))   
    #         net = MlpNetwork(fc_layers=fc_layers, inp_dim = self.obs_dim, out_dim = 2 * self.act_dim)
    #         self.current_model.append(net)
    #         self.param_counts.append(self.get_params(fc_layers))
    #         self.capacities.append(get_capacity(fc_layers, self.obs_dim, self.act_dim))
    #         shape_ind = torch.stack(shape_ind).view(-1,1)
    #         shape_inds.append(shape_ind)
    #     shape_inds = torch.cat(shape_inds) 
    #     _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = shape_inds)


    def get_params(self, net):
        ct = 0
        ct += ((self.obs_dim + 1) *net[0])
        for i in range(len(net)-1):
            ct += ((net[i] + 1) * net[i+1])
        ct += ((net[-1] +1) * 2 * self.act_dim)
        return ct            


    # def re_query_uniform_weights2(self, repeat_sample = False):
    #     self.current_model = []
    #     shape_inds = []
    #     self.create_random_net(repeat_sample)
    #     for i in range(self.meta_batch_size):
    #         net = MlpNetwork(**self.net_args[i])
    #         self.current_model.append(net)
    #         self.layer_1_actual = 0
    #         self.layer_2_actual = 0
    #         self.layer_3_actual = 0
    #         self.layer_4_actual = 0

    #         shape_ind = [[0]]
    #         for j, layer in enumerate(self.net_args[i]['fc_layers']):
    #             if j == 0:
    #                 self.layer_1_actual = layer
    #             if j == 1:
    #                 self.layer_2_actual = layer
    #             if j == 2:
    #                 self.layer_3_actual = layer
    #             if j == 3:
    #                 self.layer_4_actual = layer
    #             shape_ind.append([layer])
    #             shape_ind.append([layer])
    #         shape_ind.append([self.net_args[i]['out_dim']])
    #         shape_ind.append([self.net_args[i]['out_dim']])     
    #         shape_ind = torch.Tensor(shape_ind).to(self.device)     
    #         shape_inds.append(shape_ind)
    #     shape_inds = torch.cat(shape_inds)    
    #     # self.current_capacity = get_capacity(self.net_args[i]['fc_layers'], self.obs_dim, self.act_dim)
    #     # self.current_number_of_params = sum(p.numel() for p in self.current_model[0].parameters())
    #     _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = shape_inds)

    # def create_random_net(self, repeat_sample = False):
    #     if not repeat_sample:
    #         self.net_args = []
    #         for k in range(self.meta_batch_size):
    #             # num_layer = np.random.choice([1,2,3,4])
    #             # self.fc_layers = list(np.random.choice(self.list_of_allowable_layers[1:].cpu().numpy().astype(int),num_layer))
    #             self.fc_layers = list(np.random.choice(self.list_of_arcs))
    #             self.net_args.append({
    #                 'fc_layers':self.fc_layers,
    #                 'inp_dim':self.obs_dim,
    #                 'out_dim':2 * self.act_dim
    #             })


    def re_query_uniform_weights(self, repeat_sample = False):
        if not repeat_sample:
            self.sampled_indices = self.list_of_arc_indices[self.current_model_indices]
            self.sampled_shape_inds = torch.cat([self.list_of_shape_inds[index][:self.list_of_shape_inds_lenths[index]] for index in self.sampled_indices]).view(-1,1)   
            self.current_model_indices += self.meta_batch_size  
            if max(self.current_model_indices) >= len(self.list_of_arc_indices):
                self.current_model_indices = np.arange(self.meta_batch_size)
                # shuffle
                np.random.shuffle(self.list_of_arc_indices)

            self.current_model = [MlpNetwork(fc_layers=self.list_of_arcs[index], inp_dim = self.obs_dim, out_dim = 2 * self.act_dim) for index in self.sampled_indices]
            self.param_counts = [self.get_params(self.list_of_arcs[index]) for index in self.sampled_indices]
            self.capacities = [get_capacity(self.list_of_arcs[index], self.obs_dim, self.act_dim) for index in self.sampled_indices]
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)


    def change_graph(self, biased_sample = False, repeat_sample = False, state = None, eval = False):
        if biased_sample:
            self.re_query_biased_weights(state=state, eval = eval)
        else:
            self.re_query_uniform_weights(repeat_sample)

    
    


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