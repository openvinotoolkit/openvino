# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow.compat.v1 as tf


def main():
    tf.compat.v1.reset_default_graph()

    @tf.function
    def second_func(boxes, scores):
        # the second function is used to obtain the body graph with NonMaxSuppressionV5
        # only body graphs use named output ports
        selected_indices, selected_scores, valid_outputs = tf.raw_ops.NonMaxSuppressionV5(boxes=boxes, scores=scores,
                                                                                          max_output_size=50,
                                                                                          iou_threshold=0.4,
                                                                                          score_threshold=0.3,
                                                                                          soft_nms_sigma=0.1)
        return selected_indices, selected_scores, valid_outputs

    @tf.function
    def first_func(boxes, scores):
        selected_indices, selected_scores, valid_outputs = second_func(boxes, scores)

        # the post-processing part of the test model to get the single output
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
