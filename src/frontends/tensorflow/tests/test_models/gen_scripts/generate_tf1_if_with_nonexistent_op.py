# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# generate a model with TF1 If operation that contains unsupported operation
# in one condition branch
def main():
    def simple_if_function(cond, ind, y):
        def then_branch():
            const_one = tf.constant(1, dtype=tf.int32)
            add = tf.add(ind, const_one)
            output = tf.multiply(tf.cast(add, tf.float32), y)
            return output

        def else_branch():
            const_two = tf.constant(2, dtype=tf.int32)
            sub = tf.subtract(ind, const_two)
            output = tf.multiply(tf.cast(sub, tf.float32), y)
            return output

        if_output = tf.cond(cond, then_branch, else_branch)
        return if_output

    tf_if_graph = tf.function(simple_if_function)
    cond = np.random.randint(0, 2, []).astype(bool)
    ind = np.random.randint(1, 10, [3]).astype(np.int32)
    y = np.random.randint(-50, 50, [2, 3]).astype(np.float32)
    concrete_func = tf_if_graph.get_concrete_function(cond, ind, y)

    # lower_control_flow defines representation of If operation
    # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
    frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                    lower_control_flow=True)

    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    tf.io.write_graph(graph_def, os.path.join(sys.argv[1], "tf1_if_with_nonexistent_op"),
                      "tf1_if_with_nonexistent_op.pb",
                      False)

    with open(os.path.join(sys.argv[1], "tf1_if_with_nonexistent_op", "tf1_if_with_nonexistent_op.pb"),
              mode='rb') as file:
        modelContent = file.read()

    modelContent = modelContent.replace(b"AddV2", b"Rrrrr")

    with open(os.path.join(sys.argv[1], "tf1_if_with_nonexistent_op", "tf1_if_with_nonexistent_op.pb"),
              mode='wb') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()
