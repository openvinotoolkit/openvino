"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx

from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error


class FusedBatchNormTrainingCatch(MiddleReplacementPattern):
    """
    Replaces FusedBatchNorm(input, beta, gamma, mean, variance) with non-constant mean and variance,
    but with constant beta and gamma to a sub-expression consisting of a combinatin of Eltwise and Power
    layers and ScaleShift.
    """

    enabled = True
    replacement_id = "Fused_Batch_Norm_is_training_true_catcher"

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(kind='op', op='FusedBatchNorm', is_training=True))],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        raise Error('FusedBatchNorm doesn\'t support is_training=True. Node {}'.format(match['op'].id))
