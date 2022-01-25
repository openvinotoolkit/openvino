# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.data_loaders.synthetic_image_loader import SyntheticImageLoader
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine

class DataFreeEngine(SimplifiedEngine):
    def __init__(self, config, data_loader=None, metric=None):
        super().__init__(config)
        if not data_loader:
            self._data_loader = self.get_data_loader(config)
        else:
            self._data_loader = data_loader

    def get_data_loader(self, config):
        if config.data_type == 'image':
            return SyntheticImageLoader(config)

        raise NotImplementedError("Currently data-free optimization is available for Computer Vision models only")
