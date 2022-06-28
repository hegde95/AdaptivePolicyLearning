# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.




import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product as cartesian_product
from os.path import join

import h5py
import numpy as np
import tqdm
from hyper.graph import MlPGraph
from hyper.model import MlpNetwork, get_cell_ind
from hyper.utils import capacity


# python hypernets/mlp_net_generator.py train 10000 ./data
def main():
    parser = argparse.ArgumentParser(description='Generate MLPs')
    parser.add_argument('--inp_dim', type=int, required = True, help="Input dimensions to all MLPs")
    parser.add_argument('--out_dim', type=int, required = True, help="Output dimensions to all MLPs")
    parser.add_argument('--data_dir', type=str, required=True, help = "Data directory to store all the MLP graphs insider ./data")
    args = parser.parse_args()

    make_mlps(args.inp_dim, args.out_dim, args.data_dir)


def make_mlps(inp_dim, out_dim, data_dir):
    data_dir = join('./data', data_dir)

    os.makedirs("./data", exist_ok=True)

    device = 'cpu'  # no much benefit of using cuda

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    meta_data = {}
    meta_data['train'] = {'nets': [], 'meta': {}}
    op_types, op_types_back, primitives, primitives_back = {}, {}, {}, {}

    h5_file = join(data_dir, 'mlpnets_train.hdf5')
    meta_file = join(data_dir, 'mlpnets_train_meta.json')

    for f in [h5_file, meta_file]:
        if os.path.exists(h5_file):
            print('File %s already exists. The script will skip now to avoid accidental overwriting of the file.' % f)
            return

    with h5py.File(h5_file, 'w') as h5_data:

        h5_data.attrs['title'] = 'MlpNets'
        group = h5_data.create_group('train')
        list_of_net_args = [{'fc_layers': [0],'inp_dim': inp_dim,'out_dim': out_dim,}] #len 1 fc layers is counted as just input-output

        for num_layers in [1,2,3,4]:
            for fc_layers in cartesian_product([4, 8, 16, 32, 64, 128, 256, 512], repeat = num_layers):
                net_args = {
                            'fc_layers': fc_layers,  # number of fully connected layers before classification                        
                            'inp_dim': inp_dim,
                            'out_dim': out_dim,
                            }  
                list_of_net_args.append(net_args)                       

        for k in tqdm.tqdm(range(len(list_of_net_args)), desc='Generating networks'):
            net_args = list_of_net_args[k]
            k += 1

            graph = None
            num_params = {}

            model = MlpNetwork(**net_args).to(device)                    

            c, n = capacity(model)
            num_params['custom'] = n

            graph = MlPGraph(model, ve_cutoff=0, list_all_nodes=True)
            layers = 1

            cell_ind, n_nodes, nodes_array = 0, 0, []

            for j in range(layers):
                n_nodes += len(graph.node_info[j])

                for node in graph.node_info[j]:
                    param_name, name, sz = node[1:4]
                    cell_ind_ = get_cell_ind(param_name, layers)
                    if cell_ind_ is not None:
                        cell_ind = cell_ind_

                    assert cell_ind == j, (cell_ind, j, node)

                    if name == 'conv' and (len(sz) == 2 or sz[2] == sz[3] == 1):
                        name = 'conv_1x1'

                    if name not in primitives:
                        ind = len(primitives)
                        primitives[name] = ind
                        primitives_back[ind] = name

                    if param_name.startswith('cells.'):
                        # remove cells.x. prefix
                        pos1 = param_name.find('.')
                        assert param_name[pos1 + 1:].find('.') >= 0, node
                        pos2 = pos1 + param_name[pos1 + 1:].find('.') + 2
                        param_name = param_name[pos2:]

                    if param_name not in op_types:
                        ind = len(op_types)
                        op_types[param_name] = ind
                        op_types_back[ind] = param_name

                    nodes_array.append([primitives[name], cell_ind, op_types[param_name]])

            nodes_array = np.array(nodes_array).astype(np.uint16)

            A = graph._Adj.cpu().numpy().astype(np.uint8)
            assert nodes_array.shape[0] == n_nodes == A.shape[0] == graph.n_nodes, (nodes_array.shape, n_nodes, A.shape, graph.n_nodes)

            idx = len(meta_data['train']['nets'])
            group.create_dataset(str(idx) + '/adj', data=A)
            group.create_dataset(str(idx) + '/nodes', data=nodes_array)

            net_args['num_nodes'] = int(A.shape[0])
            net_args['num_params'] = num_params

            # net_args['genotype'] = to_dict(net_args['genotype'])
            meta_data['train']['nets'].append(net_args)
            meta_data['train']['meta']['primitives_ext'] = primitives_back
            meta_data['train']['meta']['unique_op_names'] = op_types_back

    with open(meta_file, 'w') as f:
        json.dump(meta_data, f)

    print('saved to %s and %s' % (h5_file, meta_file))

    print('\ndone')


if __name__ == '__main__':
    main()
