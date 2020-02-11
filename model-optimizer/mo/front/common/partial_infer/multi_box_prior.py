"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from mo.graph.graph import Node


def multi_box_prior_infer_mxnet(node: Node):
    img_shape = node.in_node(1).shape
    data_shape = node.in_node(0).shape
    num_ratios = len(node.aspect_ratio)

    data_H = data_shape[2] if not node.has_and_set('V10_infer') else node.in_node(0).value[0]
    data_W = data_shape[3] if not node.has_and_set('V10_infer') else node.in_node(0).value[1]

    img_H = img_shape[2] if not node.has_and_set('V10_infer') else node.in_node(1).value[0]

    if not node.has_and_set('stop_attr_upd'):
        if node.step != -1:
            node.step = img_H * node.step
        else:
            node.step = img_H / data_H
        node.min_size = [ms * img_H for ms in node.min_size]

    node['stop_attr_upd'] = True
    num_priors = len(node.min_size) + num_ratios - 1

    if node.has_and_set('V10_infer'):
        node.out_node(0).shape = np.array([2, data_H * data_W * num_priors * 4], dtype=np.int64)
    else:
        node.out_node(0).shape = np.array([1, 2, data_H * data_W * num_priors * 4], dtype=np.int64)
