# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import logging as log
import os
import tempfile
from distutils.version import LooseVersion
from pathlib import Path

from utils.path_utils import resolve_dir_path
from e2e_tests.common.ref_collector.provider import ClassProvider
from .score_tf import ScoreTensorFlowBase

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ScoreTensorFlow(ClassProvider):
    __action_name__ = "score_tf_v2"

    def __init__(self, config):
        self.saved_model_dir = resolve_dir_path(config["saved_model_dir"], as_str=True)
        self.inputs = config["inputs"]
        self.res = {}

    def get_refs(self):
        import tensorflow as tf
        """Return TensorFlow model reference results."""
        input_data_constants = {name: tf.constant(val) for name, val in self.inputs.items()}
        log.info("Running inference with tensorflow {} ...".format(tf.__version__))
        model = tf.saved_model.load(self.saved_model_dir)
        infer_func = model.signatures["serving_default"]
        self.res = infer_func(**input_data_constants)
        tf.keras.backend.clear_session()
        del model, input_data_constants, infer_func
        gc.collect()
        return self.res


class ScoreTensorFlowV2ByV1(ScoreTensorFlowBase):
    __action_name__ = "score_convert_TF2_to_TF1"

    def __init__(self, config):
        self.model = config["model"]
        super().__init__(config=config)

    def load_graph(self):
        import tensorflow as tf
        import tensorflow.compat.v1 as tf_v1
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        assert LooseVersion(tf.__version__) >= LooseVersion("2"), "This collector can't be used with TF 1.* version"

        # disable eager execution of TensorFlow 2 environment immediately
        tf_v1.disable_eager_execution()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_path = Path(tmpdir, "saved_model.pb")

            # Convert TF2 to TF1
            tf_v1.enable_eager_execution()
            imported = tf.saved_model.load(self.model)
            frozen_func = convert_variables_to_constants_v2(imported.signatures['serving_default'],
                                                            lower_control_flow=False)
            graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
            tf_v1.io.write_graph(graph_def, str(tmp_model_path.parent), tmp_model_path.name, as_text=False)

            graph = tf_v1.Graph()

            with tf_v1.gfile.GFile(str(tmp_model_path), 'rb') as f:
                graph_def = tf_v1.GraphDef()
                graph_def.ParseFromString(f.read())

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        tf_v1.disable_eager_execution()

        return graph
