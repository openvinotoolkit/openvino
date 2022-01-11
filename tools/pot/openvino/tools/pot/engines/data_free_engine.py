# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.openvino.tools.pot.data_loaders.syntetic_image_loader import SynteticImageLoader
from openvino.tools.pot.openvino.tools.pot.engines.simplified_engine import SimplifiedEngine

class DataFreeEngine(SimplifiedEngine):
    def __init__(self, config):
        super().__init__(config)
        data_loader_config = {
            'layout': config.get('layout', None),
            'data_source': config.dataset_dir,
            'subset_size': config.subset_size,
            'shape': config.get('input_shape', None),
            'generate_data': config.get('generate_data', False)
        }
        self.data_type = config.get('data_type', 'image')
        self.data_loader = self.get_data_loader(data_loader_config)

    def get_data_loader(self, config):
        if self.data_type == 'image':
            return SynteticImageLoader(config)

        raise NotImplementedError
