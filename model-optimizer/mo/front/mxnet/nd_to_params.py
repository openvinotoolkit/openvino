"""
 Copyright (c) 2017-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

import mxnet as mx
from mo.front.mxnet.extractors.utils import load_params


def save_params_file(model_name: str, args: dict, auxs: dict, iteration_number: int = 0):
    pretrained = {}
    for key in args:
        pretrained["arg:" + key] = args[key]

    for key in auxs:
        pretrained["aux:" + key] = auxs[key]

    save_model_path = '{}-{:04}.params'.format(model_name, iteration_number)
    save_model_path = os.path.expanduser(save_model_path)
    if os.path.isfile(save_model_path):
        os.remove(save_model_path)
    mx.nd.save(save_model_path, pretrained)


def add_pretrained_model(pretrained_params: dict, args: dict, pretrained_model: str, iteration_number: int,
                         input_names: str):
    if input_names:
        input_names = input_names.split(',')
    else:
        input_names = 'data'

    arg_dict = args
    if pretrained_params:
        symbol, arg_params, aux_params = mx.model.load_checkpoint(pretrained_model, iteration_number)
        arg_names = symbol.list_arguments()
        arg_dict = {}

        for name in arg_names:
            if name in input_names:
                continue
            key = "arg:" + name
            if key in pretrained_params:
                arg_dict[name] = pretrained_params[key].copyto(mx.cpu())
        del pretrained_params
        arg_dict.update(args)
    return arg_dict


def build_params_file(nd_prefix_name: str = '', pretrained_model: str = '', input_names: str = ''):
    path_wo_ext = '.'.join(pretrained_model.split('.')[:-1])
    pretrained_model_name_w_iter = path_wo_ext.split(os.sep)[-1]
    pretrained_model_name = '-'.join(path_wo_ext.split('-')[:-1])
    iteration_number = int(pretrained_model_name_w_iter.split('-')[-1])
    files_dir = os.path.dirname(pretrained_model)

    if input_names:
        model_params = load_params(pretrained_model, data_names=input_names.split(','))
    else:
        model_params = load_params(pretrained_model)

    pretrained_params = mx.nd.load(pretrained_model) if pretrained_model_name else None
    nd_args = mx.nd.load(os.path.join(files_dir, '%s_args.nd' % nd_prefix_name)) if nd_prefix_name else None
    nd_auxs = mx.nd.load(os.path.join(files_dir, '%s_auxs.nd' % nd_prefix_name)) if nd_prefix_name else None
    nd_args = add_pretrained_model(pretrained_params, nd_args, pretrained_model_name,
                                   iteration_number,
                                   input_names)

    model_params._arg_params = nd_args
    model_params._aux_params = nd_auxs
    model_params._param_names = list(nd_args.keys())
    model_params._aux_names = list(nd_auxs.keys())
    return model_params
