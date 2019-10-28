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

import logging as log
import numpy as np

from mo.graph.graph import Node, Graph
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

SUPPORTED_DATA_TYPES = {
    'float': (np.float32, 'FP32'),
    'half': (np.float16, 'FP16'),
    'FP32': (np.float32, 'FP32'),
    'FP16': (np.float16, 'FP16'),
    'I32': (np.int32, 'I32'),
    'uint8': (np.uint8, 'UI8'),
    'int32': (np.int32, 'I32'),
    'int64': (np.int64, 'I64'),
    'bool': (np.bool, 'BOOL'),
}


def data_type_str_to_np(data_type_str: str):
    return SUPPORTED_DATA_TYPES[data_type_str][0] if data_type_str in SUPPORTED_DATA_TYPES else None


def data_type_str_to_precision(data_type_str: str):
    return SUPPORTED_DATA_TYPES[data_type_str][1] if data_type_str in SUPPORTED_DATA_TYPES else None


def np_data_type_to_precision(np_data_type):
    for np_t, precision in SUPPORTED_DATA_TYPES.values():
        if np_t == np_data_type:
            return precision
    raise Error('Data type "{}" is not supported'.format(np_data_type))


def convert_blob(graph: Graph, node: Node, data_type: type, force_precision: str):
    out_edges = graph.out_edges(node.node, data=True)

    # if the data.value is used as binary weights
    if any('bin' in d for _, __, d in out_edges):
        blob = node.value
        if blob.dtype != data_type:
            # check that forcing precision to int data types does not lead to rounding
            if force_precision is not None and force_precision in ('int32', 'int64'):
                if not np.array_equal(blob, blob.astype(dtype=data_type, casting="unsafe")):
                    raise Error('The conversion of blob with value "{}" to data_type "{}" results in rounding'
                                ''.format(blob, force_precision))
            new_blob = blob.astype(dtype=data_type, casting="unsafe")
            consumers = [x.name if x.has_valid('name') else '<NO NAME>' for x in node.out_nodes()]
            log.debug(
                'Blob was converted to {} while dumping to the bin file. This blob is an input for {} nodes.'.format(
                    data_type, consumers))
            finite_match = (np.isfinite(blob) != np.isfinite(new_blob))
            zero_match = ((blob == 0) != (new_blob == 0))
            finite_match_count = np.count_nonzero(finite_match)
            zero_match_count = np.count_nonzero(zero_match)
            if finite_match_count:
                log.error(
                    ("{} elements of {} were clipped to infinity while converting a blob for " \
                     "node [{}] to {}. " +
                     refer_to_faq_msg(76)).format(
                        finite_match_count, blob.size, consumers, data_type))
            if zero_match_count:
                log.warning(
                    ("{} elements of {} were clipped to zero while converting a blob for node " \
                     " [{}] to {}. " +
                     refer_to_faq_msg(77)).format(
                        zero_match_count, blob.size, consumers, data_type))
            node.value = new_blob


def convert(graph: Graph, data_type_str: str):
    for node_name, node_attrs in graph.nodes(data=True):
        node = Node(graph, node_name)
        # if the data type is forcibly set then use it
        force_precision = None
        if node.has_valid('force_precision'):
            real_data_type_str = node_attrs['force_precision']
            force_precision = real_data_type_str
        else:
            real_data_type_str = data_type_str
        node_attrs['precision'] = data_type_str_to_precision(real_data_type_str)
        if node.kind == 'data' and node.value is not None:
            try:
                convert_blob(graph, node, data_type_str_to_np(real_data_type_str), force_precision)
            except Exception as e:
                raise Error('Coudn\'t convert blob {}, details: {}', node.soft_get('name'), e) from e
