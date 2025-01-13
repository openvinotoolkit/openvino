# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestFakeQuantize(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict, kwargs):
        assert len(inputs_dict) == 1
        assert np.array(list(inputs_dict.values())[0]) == np.array([11])
        assert 'nudged_il' in kwargs and kwargs['nudged_il'] is not None
        assert 'nudged_ih' in kwargs and kwargs['nudged_ih'] is not None
        assert 'expected_step' in kwargs and kwargs['expected_step'] is not None

        expected_nudged_input_min = kwargs['nudged_il']
        expected_nudged_input_max = kwargs['nudged_ih']
        expected_step = kwargs['expected_step']

        return {list(inputs_dict.keys())[0]: np.array([
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ])}

    def create_fake_quantize_net(self, il, ih, num_bits, narrow_range, nudged_il, nudged_ih,
                                 expected_step, ir_version, use_legacy_frontend):
        # original tf model
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(tf.float32, [11], 'parameter')
            input_min = tf.constant(il, name='input_min')
            input_max = tf.constant(ih, name='input_max')
            tf.quantization.fake_quant_with_min_max_vars(data, input_min, input_max, num_bits,
                                                         narrow_range, 'fq')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # reference graph to compare with IR
        ref_net = None

        return tf_net, ref_net

    test_data = [
        # with8BitsNoScalingNoNudging
        pytest.param(dict(il=0.0, ih=255.0, num_bits=8, narrow_range=False, nudged_il=0.0, nudged_ih=255.0,
                          expected_step=1.0), marks=pytest.mark.precommit),
        # with8BitsScalingAndNudgingDown
        dict(il=0.5, ih=128.0, num_bits=8, narrow_range=False, nudged_il=0.0, nudged_ih=127.5,
             expected_step=0.5),
        # with8BitsScalingAndNudgingUp
        dict(il=-128.0, ih=-0.5, num_bits=8, narrow_range=False, nudged_il=-127.5, nudged_ih=0.0,
             expected_step=0.5),
        # with8BitsScalingAndNudgingBetween
        dict(il=-0.1, ih=127.4, num_bits=8, narrow_range=False, nudged_il=0.0, nudged_ih=127.5,
             expected_step=0.5),
        # with8BitsNarrowRangeNoScalingNoNudging
        dict(il=0.0, ih=254.0, num_bits=8, narrow_range=True, nudged_il=0.0, nudged_ih=254.0,
             expected_step=1.0),
        # with8BitsNarrowRangeScalingAndNudgingDown
        dict(il=0.1, ih=127.1, num_bits=8, narrow_range=True, nudged_il=0.0, nudged_ih=127.0,
             expected_step=0.5),
        # with8BitsNarrowRangeScalingAndNudgingUp
        dict(il=-127.1, ih=-0.1, num_bits=8, narrow_range=True, nudged_il=-127.0, nudged_ih=0.0,
             expected_step=0.5),
        # with8BitsNarrowRangeScalingAndNudgingBetween
        dict(il=-0.1, ih=126.9, num_bits=8, narrow_range=True, nudged_il=0.0, nudged_ih=127.0,
             expected_step=0.5),
        # with7BitsNoScalingNoNudging
        dict(il=0.0, ih=127.0, num_bits=7, narrow_range=False, nudged_il=0.0, nudged_ih=127.0,
             expected_step=1.0),
        # with7BitsScalingAndNudgingDown
        dict(il=0.5, ih=64.0, num_bits=7, narrow_range=False, nudged_il=0.0, nudged_ih=63.5,
             expected_step=0.5),
        # with7BitsScalingAndNudgingUp
        dict(il=-64.0, ih=-0.5, num_bits=7, narrow_range=False, nudged_il=-63.5, nudged_ih=0.0,
             expected_step=0.5),
        # with7BitsScalingAndNudgingBetween
        dict(il=-0.1, ih=63.4, num_bits=7, narrow_range=False, nudged_il=0.0, nudged_ih=63.5,
             expected_step=0.5),
        # with7BitsNarrowRangeNoScalingNoNudging
        dict(il=0.0, ih=126.0, num_bits=7, narrow_range=True, nudged_il=0.0, nudged_ih=126.0,
             expected_step=1.0),
        # with7BitsNarrowRangeScalingAndNudgingDown
        dict(il=0.1, ih=63.1, num_bits=7, narrow_range=True, nudged_il=0.0, nudged_ih=63.0,
             expected_step=0.5),
        # with7BitsNarrowRangeScalingAndNudgingUp
        dict(il=-63.1, ih=-0.1, num_bits=7, narrow_range=True, nudged_il=-63.0, nudged_ih=0.0,
             expected_step=0.5),
        # with7BitsNarrowRangeScalingAndNudgingBetween
        dict(il=-0.1, ih=62.9, num_bits=7, narrow_range=True, nudged_il=0.0, nudged_ih=63.0,
             expected_step=0.5)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_fake_quantize(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(*self.create_fake_quantize_net(**params, ir_version=ir_version,
                                                  use_legacy_frontend=use_legacy_frontend), ie_device,
                   precision, ir_version,
                   kwargs_to_prepare_input=params, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
