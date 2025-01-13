# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import logging as log
import tensorflow as tf
from .tf_hub_ref_provider import ClassProvider


os.environ['GLOG_minloglevel'] = '3'


class ScoreTFHub(ClassProvider):
    """Reference collector for TensorFlow Hub models."""
    __action_name__ = "score_tf_hub"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.res = {}

    def get_refs(self, passthrough_data):
        inputs = passthrough_data['feed_dict']
        model = passthrough_data['model_obj']
        # repack input dictionary to tensorflow constants
        tf_inputs = {}
        for input_name, input_value in inputs.items():
            tf_inputs[input_name] = tf.constant(input_value)

        for out_name, out_value in model(**tf_inputs).items():
            self.res[out_name] = out_value.numpy()

        return self.res

