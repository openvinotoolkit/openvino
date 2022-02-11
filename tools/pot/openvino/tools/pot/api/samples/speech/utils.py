# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine


class ArkEngine(SimplifiedEngine):
    def _fill_input(self, model, image_batch):
        if 'input_names' in self.data_loader.config:
            model_inputs = {n.get_node().friendly_name: n for n in model.inputs}
            feed_dict = {}
            for input_name in self.data_loader.config['input_names']:
                input_blob = model_inputs[input_name]
                input_blob_name = self._get_input_any_name(input_blob)
                input_blob_shape = list(input_blob.shape)
                feed_dict[input_blob_name] = np.reshape(image_batch[0][input_name], input_blob_shape)
            return feed_dict
        raise Exception('input_names is not provided!')
