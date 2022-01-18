# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine


class ArkEngine(SimplifiedEngine):
    def _fill_input(self, model, image_batch):
        if 'input_names' in self.data_loader.config:
            feed_dict = {}
            for input_name in self.data_loader.config['input_names']:
                feed_dict[input_name] = image_batch[0][input_name]
            return feed_dict
        raise Exception('input_names is not provided!')
