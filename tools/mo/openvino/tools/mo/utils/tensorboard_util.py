# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# do not print INFO and WARNING messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow.compat.v1 as tf_v1
    # disable eager execution of TensorFlow 2 environment immediately
    tf_v1.disable_eager_execution()
except ImportError:
    import tensorflow as tf_v1

#in some environment suppressing through TF_CPP_MIN_LOG_LEVEL does not work
tf_v1.get_logger().setLevel("ERROR")

try:
    import tensorflow.contrib  # pylint: disable=no-name-in-module,import-error
except:
    pass  # we try to import contrib for loading models that use contrib operations
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def dump_for_tensorboard(graph_def: tf_v1.GraphDef, logdir: str):
    try:
        # TODO: graph_def is a deprecated argument, use graph instead
        print('Writing an event file for the tensorboard...')
        with tf_v1.summary.FileWriter(logdir=logdir, graph_def=graph_def) as writer:
            writer.flush()
        print('Done writing an event file.')
    except Exception as err:
        raise Error('Cannot write an event file for the tensorboard to directory "{}". ' +
                    refer_to_faq_msg(36), logdir) from err
