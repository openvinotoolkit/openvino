# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow.compat.v1 as tf


def main():
    tf.compat.v1.reset_default_graph()

    @tf.function
    def second_func(boxes, scores):
        const_one = tf.constant(1, dtype=tf.float32)
        const_two = tf.constant(2, dtype=tf.float32)
        const_five = tf.constant(5, dtype=tf.float32)
        const_seven = tf.constant(5, dtype=tf.int32)
        const_ten = tf.constant(10, dtype=tf.int32)

        boxes = tf.raw_ops.Mul(x=boxes, y=const_two)
        scores = tf.raw_ops.Add(x=scores, y=const_one)
        selected_indices, selected_scores, valid_outputs = tf.raw_ops.NonMaxSuppressionV5(boxes=boxes, scores=scores,
                                                                                          max_output_size=50,
                                                                                          iou_threshold=0.4,
                                                                                          score_threshold=0.3,
                                                                                          soft_nms_sigma=0.1)
        selected_indices = tf.raw_ops.Add(x=selected_indices, y=const_ten)
        selected_scores = tf.raw_ops.Mul(x=selected_scores, y=const_five)
        valid_outputs = tf.raw_ops.Add(x=valid_outputs, y=const_seven)

        return selected_indices, selected_scores, valid_outputs

    @tf.function
    def first_func(boxes, scores):
        selected_indices, selected_scores, valid_outputs = second_func(boxes, scores)

        selected_indices = tf.raw_ops.Reshape(tensor=selected_indices, shape=[-1])
        selected_scores = tf.raw_ops.Cast(x=selected_scores, DstT=tf.int32)
        selected_scores = tf.raw_ops.Reshape(tensor=selected_scores, shape=[-1])
        valid_outputs = tf.raw_ops.Reshape(tensor=valid_outputs, shape=[-1])
        concat = tf.raw_ops.Concat(concat_dim=0, values=[selected_indices, selected_scores, valid_outputs])
        return concat

    tf_net = first_func.get_concrete_function(
        tf.constant([[0.1, 0.2, 0.3, 0.5], [0.4, 0.2, 0.1, 0.8]], dtype=tf.float32),
        tf.constant([0, 2], dtype=tf.float32)).graph.as_graph_def()

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "nms_named_outputs"),
                      "nms_named_outputs.pb", False)


if __name__ == "__main__":
    main()
