# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
import unittest
from pathlib import Path

import pytest
import tensorflow as tf
from openvino.frontend import OpConversionFailure

from common import constants
from common.utils.tf_utils import save_to_pb


@tf.function(
    input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def then_branch(input):
    tf.raw_ops.WriteFile(filename="out_tensor1.txt", contents="abc")
    return tf.constant(0, dtype=tf.float32)


@tf.function(
    input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def else_branch(input):
    tf.raw_ops.WriteFile(filename="out_tensor2.txt", contents="cdf")
    return tf.constant(0, dtype=tf.float32)


def create_if_net(path):
    #  tf.compat.v1.reset_default_graph()
    # Create the graph and model
    with tf.compat.v1.Session() as sess:
        cond_inp = tf.compat.v1.placeholder(tf.float32, [], 'value')
        input = tf.compat.v1.placeholder(tf.float32, [], 'value')
        if_op = tf.raw_ops.If(cond=tf.raw_ops.Less(x=cond_inp, y=tf.constant(0, dtype=tf.float32)),
                              input=[input],
                              Tout=[tf.float32],
                              then_branch=then_branch.get_concrete_function(),
                              else_branch=else_branch.get_concrete_function())
        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph
    return save_to_pb(tf_net, path)


def create_part_call_net(path):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32),
                         tf.TensorSpec(shape=[], dtype=tf.float32)])
    def part_call_func(input, cond_inp):
        if_op = tf.raw_ops.If(cond=tf.raw_ops.Less(x=cond_inp, y=tf.constant(0, dtype=tf.float32)),
                              input=[input],
                              Tout=[tf.float32],
                              then_branch=then_branch.get_concrete_function(),
                              else_branch=else_branch.get_concrete_function())
        return tf.constant(0, dtype=tf.float32)

    tf.compat.v1.reset_default_graph()
    # Create the graph and model
    with tf.compat.v1.Session() as sess:
        input = tf.compat.v1.placeholder(tf.float32, [], 'value')
        cond_input = tf.compat.v1.placeholder(tf.float32, [], 'value')
        tf.raw_ops.PartitionedCall(args=[input, cond_input],
                                   Tout=[tf.float32],
                                   f=part_call_func.get_concrete_function())
        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph
    return save_to_pb(tf_net, path)


class TestUnsupportedOps(unittest.TestCase):
    def setUp(self):
        Path(constants.out_path).mkdir(parents=True, exist_ok=True)
        test_name = re.sub(r"[^\w_]", "_", unittest.TestCase.id(self))
        self.tmp_dir = tempfile.TemporaryDirectory(dir=constants.out_path, prefix=f"{test_name}").name

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_write_file_in_if(self):
        from openvino.tools.ovc import convert_model

        model = create_if_net(self.tmp_dir)

        with self.assertRaisesRegex(OpConversionFailure,
                                    ".*Internal error, no translator found for operation\(s\): WriteFile.*"):
            convert_model(model)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_write_file_in_part_call(self):
        from openvino.tools.ovc import convert_model

        model = create_part_call_net(self.tmp_dir)

        with self.assertRaisesRegex(OpConversionFailure,
                                    '.*Internal error, no translator found for operation\(s\): WriteFile.*'):
            convert_model(model)
