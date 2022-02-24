# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.data_loaders.image_loader import ImageLoader
from openvino.tools.pot.graph.model_utils import get_nodes_by_type


def create_data_loader(config, model):
    """
    Factory to create instance of engine class based on config
    :param config: engine config section from toolkit config file
    :param model: CompressedModel instance to find out input shape
    :return: instance of DataLoader descendant class
    """

    inputs = get_nodes_by_type(model, ['Parameter'], recursively=False)

    if len(inputs) > 1 and\
            not any([tuple(i.shape) == (1, 3) for i in inputs]):
        raise RuntimeError('IEEngine supports networks with single input or net with 2 inputs. '
                           'In second case there are image input and image info input '
                           'Actual inputs number: {}'.format(len(inputs)))

    data_loader = None
    for in_node in inputs:
        if tuple(in_node.shape) != (1, 3):
            data_loader = ImageLoader(config)
            data_loader.shape = in_node.shape
            data_loader.get_layout(in_node)
            return data_loader

    if data_loader is None:
        raise RuntimeError('There is no node with image input')

    return data_loader
