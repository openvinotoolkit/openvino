# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
import numpy as np


class TestMultinomial(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict, kwargs):
        inputs_dict["num_samples:0"] = np.array(kwargs["num_samples:0"], dtype=np.int32)
        inputs_dict["probs:0"] = kwargs["input:0"]
        return inputs_dict

    def create_tf_multinomial_net_shape(
        self, global_seed, op_seed, logits_shape, input_type, out_type
    ):
        tf.compat.v1.reset_default_graph()
        # Configuration required to make multinomial randomness predictable across devices, results depends on TF parallel execution.
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        # Create the graph and model
        with tf.compat.v1.Session(config=session_conf) as sess:
            probs = tf.compat.v1.placeholder(input_type, logits_shape, "probs")
            num_samples = tf.compat.v1.placeholder(tf.int32, [], "num_samples")
            if global_seed is not None:
                tf.random.set_seed(global_seed)
            tf.raw_ops.ZerosLike(
                x=tf.raw_ops.Multinomial(
                    logits=tf.math.log(probs),
                    num_samples=num_samples,
                    seed=global_seed,
                    seed2=op_seed,
                    output_dtype=out_type,
                )
            )

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    def create_tf_multinomial_net_exact(
        self, global_seed, op_seed, logits_shape, input_type, out_type
    ):
        tf.compat.v1.reset_default_graph()
        # Configuration required to make multinomial randomness predictable across devices, results depends on TF parallel execution.
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        # Create the graph and model
        with tf.compat.v1.Session(config=session_conf) as sess:
            probs = tf.compat.v1.placeholder(input_type, logits_shape, "probs")
            num_samples = tf.compat.v1.placeholder(tf.int32, [], "num_samples")
            if global_seed is not None:
                tf.random.set_seed(global_seed)
                tf.raw_ops.Multinomial(
                    logits=tf.math.log(probs),
                    num_samples=num_samples,
                    seed=global_seed,
                    seed2=op_seed,
                    output_dtype=out_type,
                )

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("out_type", [tf.int32, tf.int64])
    @pytest.mark.parametrize(
        ("input", "num_samples", "seed", "test_type"),
        [
            (
                np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.float32),
                1024,
                [32465, 48971],
                "exact",
            ),
            (
                np.array(
                    [
                        [0.001, 0.001, 0.1, 0.9],
                        [5, 10, 1e-5, 256],
                        [1, 1e-5, 1e-5, 1e-5],
                    ],
                    dtype=np.float64,
                ),
                256,
                [32465, 48971],
                "shape",
            ),
            (np.array([[1, 1, 1, 1]], dtype=np.float16), 1024, [1, 1], "shape"),
            (
                np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 0, 0, 0]], dtype=np.float32),
                1,
                [78132, None],
                "shape",
            ),
            (
                np.array([[7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]], dtype=np.float32),
                1024,
                [32465, None],
                "shape",
            ),
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_multinomial_basic(
        self,
        input,
        num_samples,
        seed,
        out_type,
        test_type,
        ie_device,
        precision,
        ir_version,
        temp_dir,
        use_legacy_frontend,
    ):
        if ie_device == "GPU":
            pytest.skip("Multinomial is not supported on GPU")
        net = getattr(self, f"create_tf_multinomial_net_{test_type}")
        self._test(
            *net(
                global_seed=seed[0],
                op_seed=seed[1],
                logits_shape=input.shape,
                input_type=input.dtype,
                out_type=out_type,
            ),
            ie_device,
            precision,
            temp_dir=temp_dir,
            ir_version=ir_version,
            use_legacy_frontend=use_legacy_frontend,
            kwargs_to_prepare_input={"input:0": input, "num_samples:0": num_samples},
        )
