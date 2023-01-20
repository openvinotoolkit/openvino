# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from common.layer_test_class import CommonLayerTest
from common.utils.tflite_utils import get_tflite_results, save_pb_to_tflite
from common.utils.tf_utils import save_to_pb


class TFLiteLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        pb_model = save_to_pb(framework_model, save_path)
        return save_pb_to_tflite(pb_model)

    def get_framework_results(self, inputs_dict, model_path):
        return get_tflite_results(self.use_new_frontend, self.use_old_api, inputs_dict, model_path)

    @staticmethod
    def make_model(**params):
        raise RuntimeError("This is Tensorflow Lite base layer test class, "
                           "please implement make_model function for the specific test")

    def _test(self, ie_device, precision, temp_dir, **params):
        model = self.make_model(**params)
        super()._test(model, None, ie_device, precision, None, temp_dir, False, True)
