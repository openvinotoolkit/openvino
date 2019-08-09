"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import collect_until_token, read_binary_bool_token, read_binary_integer32_token, \
                                        read_binary_float_token
from extensions.ops.normalize import NormalizeOp
from mo.utils.error import Error


class NormalizeComponentFrontExtractor(FrontExtractorOp):
    op = 'normalizecomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        try:
            collect_until_token(pb, b'<Dim>')
        except Error:
            try:
                pb.seek(0)
                collect_until_token(pb, b'<InputDim>')
            except Error:
                raise Error("Neither <Dim> nor <InputDim> were found")
        in_dim = read_binary_integer32_token(pb)

        try:
            collect_until_token(pb, b'<TargetRms>')
            target_rms = read_binary_float_token(pb)
        except Error:
            # model does not contain TargetRms
            target_rms = 1.0

        try:
            collect_until_token(pb, b'<AddLogStddev>')
            add_log = read_binary_bool_token(pb)
        except Error:
            # model does not contain AddLogStddev
            add_log = False

        if add_log is not False:
            raise Error("AddLogStddev True  in Normalize component is not supported")

        scale = target_rms * np.sqrt(in_dim)

        attrs = {
                 'eps': 0.00000001,
                 'across_spatial': 0,
                 'channel_shared': 1,
                 'in_dim': in_dim,
        }
        embed_input(attrs, 1, 'weights', [scale])

        NormalizeOp.update_node_stat(node, attrs)
        return __class__.enabled
