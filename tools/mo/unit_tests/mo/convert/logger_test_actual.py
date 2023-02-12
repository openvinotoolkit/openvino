# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys

def create_tf_model():
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        relu = tf.nn.relu(inp1 + inp2, name='Relu')

        output = tf.nn.sigmoid(relu, name='Sigmoid')

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def
    return tf_net

def run_main():
    from openvino.tools.mo import convert_model

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    tf_model = create_tf_model()
    _ = convert_model(tf_model)

    log.info("test message 1")

    logger = log.getLogger()
    assert logger.level == 20
    assert len(logger.handlers) == 1
    assert len(logger.filters) == 0

    _ = convert_model(tf_model, log_level="DEBUG", silent=False)

    log.info("test message 2")

    logger = log.getLogger()
    assert logger.level == 20
    assert len(logger.handlers) == 1
    assert len(logger.filters) == 0


    _ = convert_model(tf_model, log_level="CRITICAL", silent=False)


    log.info("test message 3")

    logger = log.getLogger()
    assert logger.level == 20
    assert len(logger.handlers) == 1
    assert len(logger.filters) == 0

if __name__ == "__main__":
    run_main()
