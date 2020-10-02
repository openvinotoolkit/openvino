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
from extensions.ops.statistics import KaldiStatisticsPooling
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import read_binary_float_token, collect_until_token, read_binary_integer32_token, \
    read_binary_bool_token


class KaldiStatisticsPoolingExtractor(FrontExtractorOp):
    op = 'statisticspoolingcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<InputDim>')
        input_dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<InputPeriod>')
        input_period = read_binary_integer32_token(pb)
        collect_until_token(pb, b'<LeftContext>')
        left_context = read_binary_integer32_token(pb)
        collect_until_token(pb, b'<RightContext>')
        right_context = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<NumLogCountFeatures>')
        num_log_count_features = read_binary_integer32_token(pb)
        collect_until_token(pb, b'<OutputStddevs>')
        output_std_evs = read_binary_bool_token(pb)
        collect_until_token(pb, b'<VarianceFloor>')
        variance_floor = read_binary_float_token(pb)

        KaldiStatisticsPooling.update_node_stat(node)
        return cls.enabled
