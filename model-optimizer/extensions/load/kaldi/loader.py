"""
 Copyright (C) 2020 Intel Corporation

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
from extensions.load.loader import Loader
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import extract_node_attrs
from mo.front.kaldi.extractor import kaldi_extractor, kaldi_type_extractors
from mo.front.kaldi.loader.loader import load_kaldi_model
from mo.graph.graph import Graph
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class KaldiLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        try:
            load_kaldi_model(graph, argv.input_model)
        except Exception as e:
            raise Error('Model Optimizer is not able to parse Kaldi model {}. '.format(argv.input_model) +
                        refer_to_faq_msg(91)) from e
        graph.check_empty_graph('load_kaldi_nnet_model')
        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'kaldi'

        update_extractors_with_extensions(kaldi_type_extractors)
        extract_node_attrs(graph, lambda node: kaldi_extractor(node))
