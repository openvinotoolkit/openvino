# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.splice import Splice
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import find_next_tag, read_placeholder, read_binary_integer32_token, \
    collect_until_whitespace
from mo.front.kaldi.utils import read_binary_vector
from mo.utils.error import Error


class SpliceFrontExtractor(FrontExtractorOp):
    op = 'splicecomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        mapping_rule = {
            'context': list()
        }
        tag = find_next_tag(pb)
        if tag == '<LeftContext>':
            read_placeholder(pb, 1)
            l_context = read_binary_integer32_token(pb)
            tag = find_next_tag(pb)
            if tag != '<RightContext>':
                raise Error('Unknown token {} in SpliceComponent node {}'.format(tag, node.id))
            read_placeholder(pb, 1)
            r_context = read_binary_integer32_token(pb)
            for i in range(-l_context, r_context + 1):
                mapping_rule['context'].append(i)
        elif tag == '<Context>':
            collect_until_whitespace(pb)
            mapping_rule['context'] = read_binary_vector(pb, False, dtype=np.int32)
        else:
            raise Error('Unknown token {} in SpliceComponent node {}'.format(tag, node.id))

        tag = find_next_tag(pb)
        if tag == '<ConstComponentDim>':
            read_placeholder(pb, 1)
            const_dim = read_binary_integer32_token(pb)
            mapping_rule['const_dim'] = const_dim

        Splice.update_node_stat(node, mapping_rule)
        return cls.enabled
