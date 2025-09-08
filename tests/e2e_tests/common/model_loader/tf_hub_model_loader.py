# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
import tensorflow_hub as hub

from e2e_tests.common.model_loader.provider import ClassProvider


class TFHubModelLoader(ClassProvider):
    """TFHub models loader runner."""
    __action_name__ = "load_tf_hub_model"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self._config = config
        self.prepared_model = None

    def load_model(self, input_data):
        model_name = self._config['model_name']
        model_link = self._config['model_link']
        load = hub.load(model_link)
        if 'serving_default' in list(load.signatures.keys()):
            self.prepared_model = load.signatures['serving_default']
        elif 'default' in list(load.signatures.keys()):
            self.prepared_model = load.signatures['default']
        else:
            signature_keys = sorted(list(load.signatures.keys()))
            assert len(signature_keys) > 0, "No signatures for a model {}, url {}".format(model_name, model_link)
            self.prepared_model = load.signatures[signature_keys[0]]
        self.prepared_model._backref_to_saved_model = load
        return self.prepared_model

