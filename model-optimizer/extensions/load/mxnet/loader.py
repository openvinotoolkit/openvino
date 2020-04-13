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

from mo.utils.error import FrameworkError, Error
from mo.utils.utils import refer_to_faq_msg

try:
    import mxnet
except ImportError:
    raise Error('Module mxnet was not found. Please install appropriate version of mxnet via install_prerequisites '
                'script.' + refer_to_faq_msg(52))

from extensions.load.loader import Loader
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import extract_node_attrs
from mo.front.mxnet.extractor import mxnet_op_extractors, mxnet_op_extractor
from mo.front.mxnet.loader import symbol2nx, load_symbol_def
from mo.front.mxnet.nd_to_params import save_params_file
from mo.graph.graph import Graph


class MxNetLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        try:
            model_nodes, model_params, model_name, iteration_number = load_symbol_def(argv.input_model,
                                                                                      argv.input_symbol,
                                                                                      argv.input,
                                                                                      argv.nd_prefix_name,
                                                                                      argv.pretrained_model_name,
                                                                                      argv.legacy_mxnet_model)
        except (ValueError, mxnet.base.MXNetError) as e:
            raise FrameworkError(
                'The following error happened while loading mxnet model {}: {}. ' +
                refer_to_faq_msg(53),
                argv.input_model,
                str(e)
            ) from e

        if argv.nd_prefix_name and argv.pretrained_model_name and argv.save_params_from_nd:
            save_params_file(model_name, model_params._arg_params, model_params._aux_params, iteration_number)

        update_extractors_with_extensions(mxnet_op_extractors)
        symbol2nx(graph, model_nodes, model_params, argv.input)
        graph.check_empty_graph('symbol2nx. It may happen due to problems with loaded model')

        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'mxnet'
        graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3

        extract_node_attrs(graph, mxnet_op_extractor)
