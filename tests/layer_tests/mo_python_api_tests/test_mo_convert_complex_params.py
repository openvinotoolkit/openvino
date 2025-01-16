# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
from openvino.runtime import Model, Layout, PartialShape, Shape, layout_helpers, Type, Dimension
from openvino.tools.mo import LayoutMap, InputCutInfo
import openvino.runtime as ov
from common.mo_convert_test_class import CommonMOConvertTest
from common.utils.tf_utils import save_to_pb


class TestComplexParams(CommonMOConvertTest):
    def create_tf_model(self, tmp_dir):
        #
        #   Create Tensorflow model with multiple inputs/outputs
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp1 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input1')
            inp2 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input2')
            inp3 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input3')

            relu1 = tf.nn.relu(inp1, name='Relu1')
            relu2 = tf.nn.relu(inp2, name='Relu2')
            relu3 = tf.nn.relu(inp3, name='Relu3')

            concat = tf.concat([relu1, relu2, relu3], axis=0, name='Concat')

            outputs = tf.split(concat, 3)
            outputs_list = []
            for i, output in enumerate(outputs):
                outputs_list.append(tf.nn.sigmoid(output, name='Sigmoid_{}'.format(i)))

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    def create_tf_model_no_concat(self, tmp_dir):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp1 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input1')
            inp2 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input2')
            inp3 = tf.compat.v1.placeholder(tf.bool, [], 'Input3')
            output2 = inp3

            relu1 = tf.nn.sigmoid(inp1, name='Relu1')
            relu2 = tf.nn.sigmoid(inp2, name='Relu2')
            output = relu1 + relu2

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    def create_tf_model_single_input_output(self, tmp_dir):
        #
        #   Create Tensorflow model with single input/output
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input')

            relu = tf.nn.relu(inp, name='Relu')

            output = tf.nn.sigmoid(relu, name='Sigmoid')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    def create_tf_model_no_sigmoid(self, tmp_dir):
        #
        #   Create Tensorflow model without Sigmoid nodes
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    def create_tf_param_res_model(self, tmp_dir):
        #
        #   Create Tensorflow model with following pattern:
        #   Input ---\
        #                 Add --> Identity
        #   Input1 ---/
        #
        #   This graph is needed for transform test. Input and Identity are replaced with ReadValue and Assign ops.

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input')
            inp1 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input1')
            sum1 = tf.add(inp, inp1, "Add1")
            result = tf.identity(sum1, name='Identity')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    test_data = [
        {'params_test': {'input_shape': [PartialShape([2, 3, 4]),
                                         [2, 3, 4],
                                         [Dimension(2), Dimension(3), Dimension(4)]],
                         'input':['Input1', 'Input2', 'Relu3'], 'compress_to_fp16': True},
         'params_ref': {'input_shape': "[2,3,4],[2,3,4],[2,3,4]", 'input': 'Input1,Input2,Relu3'}},
        {'params_test': {'input_shape': [PartialShape([Dimension(), Dimension(1, 3), Dimension(4, -1), Dimension(-1, 5)]),
                                         [Dimension(), Dimension(1, 3), 4, Dimension(-1, 5)],
                                         [Dimension(), 3, Dimension(4, -1), Dimension(-1, 5)]],
                         'compress_to_fp16': True,
                         'input':['Input1', 'Input2', 'Relu3']},
         'params_ref': {'input_shape': "[?,1..3,4..,..5],[?,1..3,4,..5],[?,3,4..,..5]", 'input': 'Input1,Input2,Relu3'}},
        {'params_test': {'input': [InputCutInfo("Relu1", Shape([3, 2]), Type(np.int32)),
                                   InputCutInfo("Relu2", PartialShape([Dimension(3, 10), Dimension(2, -1)]), np.int32),
                                   InputCutInfo("Relu3", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6])]},
         'params_ref': {'input': "Relu1[3 2]{i32},Relu2[3..10 2..]{i32},Relu3[3 2]{i32}->[1 2 3 4 5 6]"}},
        {'params_test': {'input': [("Relu1", Shape([3, 2]), Type(np.int32)),
                                   (np.int32, "Relu2", PartialShape([Dimension(3, 10), Dimension(2, -1)])),
                                   ([3, 2],"Relu3",  Type(np.int32))]},
         'params_ref': {'input': "Relu1[3 2]{i32},Relu2[3..10 2..]{i32},Relu3[3 2]{i32}"}},
        {'params_test': {'output': ["Sigmoid_0", "Sigmoid_2"]},
         'params_ref': {'output': "Sigmoid_0,Sigmoid_2"}},
        {'params_test': {'mean_values': {'Input1:0': [0.5,1.3,0.67], 'Input2:0':[4.2, 6.7, 3.15], 'Input3:0':[0.757, 4.6, 7.3]},
                         'compress_to_fp16': True},
         'params_ref': {'mean_values': "Input1:0[0.5,1.3,0.67],Input2:0[4.2,6.7,3.15],Input3:0[0.757,4.6,7.3]"}},
        {'params_test': {
            'mean_values': [[0.5, 1.3, 0.67], [4.2, 6.7, 3.15], [0.757, 4.6, 7.3]], 'compress_to_fp16': True},
         'params_ref': {'mean_values': "[0.5,1.3,0.67],[4.2,6.7,3.15],[0.757,4.6,7.3]"}},
        {'params_test': {'scale_values': {'Input1:0': [0.5,1.3,0.67], 'Input2:0':[4.2, 6.7, 3.15], 'Input3:0':[0.757, 4.6, 7.3]},
                         'compress_to_fp16': True},
         'params_ref': {'scale_values': "Input1:0[0.5,1.3,0.67],Input2:0[4.2,6.7,3.15],Input3:0[0.757,4.6,7.3]"}},
        {'params_test': {
            'scale_values': [[0.5, 1.3, 0.67], [4.2, 6.7, 3.15], [0.757, 4.6, 7.3]], 'compress_to_fp16': True},
         'params_ref': {'scale_values': "[0.5,1.3,0.67],[4.2,6.7,3.15],[0.757,4.6,7.3]"}},
        {'params_test': {
            'source_layout': {'Input1:0': Layout("nchw"), 'Input2:0': "nchw", 'Input3:0': "nc??"}, 'compress_to_fp16': True},
         'params_ref': {'source_layout': "Input1:0(nchw),Input2:0(nchw),Input3:0(nc??)"}},
        {'params_test': {
            'target_layout': {'Input1:0': Layout("nhwc"), 'Input2:0': "nhwc", 'Input3:0': "n??c"}, 'compress_to_fp16': True},
            'params_ref': {'target_layout': "Input1:0(nhwc),Input2:0(nhwc),Input3:0(n??c)"}},
        {'params_test': {
            'layout': {'Input1:0': LayoutMap(source_layout=Layout("nchw"), target_layout="nhwc"),
                       'Input2:0': LayoutMap(source_layout="nc??", target_layout=Layout("n??c")),
                       'Input3:0': LayoutMap(source_layout="abcd", target_layout="acdb")}, 'compress_to_fp16': True},
            'params_ref': {'layout': "Input1:0(nchw->nhwc),Input2:0(nc??->n??c),Input3:0(abcd->acdb)"}},
        {'params_test': {'input': [PartialShape([2, 3, 4]), [2, 3, 4], [Dimension(2), Dimension(3), Dimension(4)]]},
         'params_ref': {'input_shape': "[2,3,4],[2,3,4],[2,3,4]", 'input': 'Input1:0,Input2:0,Input3:0'}},
        {'params_test': {'input': [np.int32, Type(np.int32), np.int32]},
         'params_ref': {'input': 'Input1:0{i32},Input2:0{i32},Input3:0{i32}'}},
        {'params_test': {'input': [InputCutInfo(shape=[1], type=np.int32, value=[10]),
                                   InputCutInfo(shape=[1], type=np.int32, value=[20]),
                                   InputCutInfo(shape=[1], type=np.int32, value=[30])]},
         'params_ref': {'input': 'Input1[1]{i32}->[10],Input2[1]{i32}->[20],Input3[1]{i32}->[30]'}}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mo_convert_tf_model(self, params, ie_device, precision, ir_version,
                                 temp_dir, use_legacy_frontend):
        tf_net_path = self.create_tf_model(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'use_convert_model_from_mo': True})
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    test_data = [
        {'params_test': {'input_shape': [[Dimension(1), 2, 3], [Dimension(1), 2, 3]],
                         'freeze_placeholder_with_value': 'Input3->[1]'},

         'params_ref': {'input_shape': '[1,2,3],[1,2,3]',
                        'freeze_placeholder_with_value': 'Input3->[1]'}},
        {'params_test': {'input': [PartialShape([Dimension(-1), 5, 6]), [-1, 5, 6]],
                         'freeze_placeholder_with_value': 'Input3->[1]'},

         'params_ref': {'input': 'Input1:0[?,5,6],Input2:0[?,5,6]',
                        'freeze_placeholder_with_value': 'Input3->[1]'}},
        {'params_test': {'input': [np.float16, np.float16],
                         'input_shape': [[10, 20], [10, 20]],
                         'freeze_placeholder_with_value': 'Input3->[1]'},

         'params_ref': {'input': 'Input1:0{f16},Input2:0{f16}',
                        'input_shape': "[10,20],[10,20]",
                        'freeze_placeholder_with_value': 'Input3->[1]'}},

    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mo_convert_tf_model_no_concat(self, params, ie_device, precision, ir_version,
                                 temp_dir, use_legacy_frontend):
        tf_net_path = self.create_tf_model_no_concat(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        test_params.update({'use_convert_model_from_mo': True, 'compress_to_fp16': True})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    test_data = [
        # When use_convert_model_from_mo=True legacy openvino.tools.mo.convert_model is used
        # By default compress_to_fp16 in Python API is False but for mo cli tool (used for params_ref) it's True.
        # compress_to_fp16 should be specified explicitly either in 'param_test' or  'params_ref' (or in both)
        # Check all args combinations.
        {'params_test': {'input_shape': PartialShape([2, 3, 4]), 'compress_to_fp16': True},
         'params_ref': {'input_shape': "[2,3,4]"}},
        {'params_test': {'input_shape': PartialShape([2, 3, 4])},
         'params_ref': {'input_shape': "[2,3,4]", 'compress_to_fp16': False}},
        {'params_test': {'input_shape': PartialShape([2, 3, 4]), 'compress_to_fp16': True},
         'params_ref': {'input_shape': "[2,3,4]", 'compress_to_fp16': True}},
        {'params_test': {'input_shape': PartialShape([2, 3, 4]), 'compress_to_fp16': False},
         'params_ref': {'input_shape': "[2,3,4]", 'compress_to_fp16': False}},

        # ovc.convert_model with save_model are used, by default save_model compresses to fp16 same as cli tool.
        # Check all args combinations.
        {'params_test': {'input': InputCutInfo("Relu", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6])},
         'params_ref': {'input': "Relu[3 2]{i32}->[1 2 3 4 5 6]"}},
        {'params_test': {'input': InputCutInfo("Relu", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6]), 'compress_to_fp16': True},
         'params_ref': {'input': "Relu[3 2]{i32}->[1 2 3 4 5 6]"}},
        {'params_test': {'input': InputCutInfo("Relu", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6])},
         'params_ref': {'input': "Relu[3 2]{i32}->[1 2 3 4 5 6]", 'compress_to_fp16': True}},
        {'params_test': {'input': InputCutInfo("Relu", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6]), 'compress_to_fp16': True},
         'params_ref': {'input': "Relu[3 2]{i32}->[1 2 3 4 5 6]", 'compress_to_fp16': True}},
        {'params_test': {'input': InputCutInfo("Relu", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6]), 'compress_to_fp16': False},
         'params_ref': {'input': "Relu[3 2]{i32}->[1 2 3 4 5 6]", 'compress_to_fp16': False}},

        {'params_test': {'input_shape': [Dimension(), Dimension(1, 3), 4, Dimension(-1, 5)], 'compress_to_fp16': True},
         'params_ref': {'input_shape': "[?,1..3,4,..5]"}},
        {'params_test': {'input': InputCutInfo("Relu", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6])},
         'params_ref': {'input': "Relu[3 2]{i32}->[1 2 3 4 5 6]"}},
        {'params_test': {'input': ("Relu", [3, 2], Type(np.int32))},
         'params_ref': {'input': "Relu[3 2]{i32}"}},
        {'params_test': {'input': ("Relu", Type(np.int32))},
         'params_ref': {'input': "Relu{i32}"}},
        {'params_test': {'input': ("Relu", [3, 2])},
         'params_ref': {'input': "Relu[3 2]"}},
        {'params_test': {'input': ("Relu")},
         'params_ref': {'input': "Relu"}},
        {'params_test': {'mean_values': [0.5, 1.3, 0.67], 'compress_to_fp16': True},
         'params_ref': {'mean_values': "[0.5,1.3,0.67]"}},
        {'params_test': {'scale_values': [0.5, 1.3, 0.67], 'compress_to_fp16': True},
         'params_ref': {'scale_values': "[0.5,1.3,0.67]"}},
        {'params_test': {'source_layout': Layout("nchw"), 'compress_to_fp16': True},
         'params_ref': {'source_layout': "nchw"}},
        {'params_test': {'target_layout': Layout("nchw"), 'compress_to_fp16': True},
         'params_ref': {'target_layout': "nchw"}},
        {'params_test': {'layout': LayoutMap(source_layout=Layout("nchw"), target_layout="nhwc"), 'compress_to_fp16': True},
         'params_ref': {'layout': "nchw->nhwc"}},
        {'params_test': {'layout': Layout("nchw"), 'compress_to_fp16': True},
         'params_ref': {'layout': "nchw"}},
        {'params_test': {'input': [3, 2]},
         'params_ref': {'input': "Input:0[3 2]"}},
        {'params_test': {'input': [Dimension(3,10), 2]},
         'params_ref': {'input': "Input:0[3..10 2]"}},
        {'params_test': {'input': (-1, 10)},
         'params_ref': {'input': "Input:0[?,10]"}},
        {'params_test': {'input': PartialShape([-1, 10])},
         'params_ref': {'input': "Input:0[?,10]"}},
        {'params_test': {'input': np.int32},
         'params_ref': {'input': "Input:0{i32}"}},
        {'params_test': {'input': InputCutInfo(shape=[1], type=np.int32, value=[10])},
         'params_ref': {'input': "Input:0[1]{i32}->[10]"}},
        {'params_test': {'input': (np.int32, [1, 2, 3])},
         'params_ref': {'input': "Input:0[1,2,3]{i32}"}},
        {'params_test': {'input_shape': [Dimension(3, 10), 10, -1], 'compress_to_fp16': True},
         'params_ref': {'input_shape': '[3..10,10,?]'}},
        {'params_test': {'input': [Dimension(3, 10), 10, -1]},
         'params_ref': {'input': 'Input:0[3..10,10,?]'}},
        {'params_test': {'input': PartialShape([1, 100, 100, 3]), 'mean_values': [0.5, 1.3, 0.67], 'compress_to_fp16': True},
         'params_ref': {'input': "Input:0[1,100,100,3]", 'mean_values': "[0.5,1.3,0.67]"}},
        {'params_test': {'input': [1, 100, 100, 3], 'scale_values': [0.5, 1.3, 0.67], 'compress_to_fp16': True},
         'params_ref': {'input': "Input:0[1,100,100,3]", 'scale_values': "[0.5,1.3,0.67]"}},
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_tf_model_single_input_output(self, params, ie_device, precision, ir_version,
                                                     temp_dir, use_legacy_frontend):
        tf_net_path = self.create_tf_model_single_input_output(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'use_convert_model_from_mo': True})
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_clearing_transformation_registry(self, ie_device, precision, ir_version,
                                                         temp_dir, use_legacy_frontend):
        tf_net_path = self.create_tf_model_single_input_output(temp_dir)
        from openvino.tools.mo import convert_model

        config_path = os.path.join(os.path.dirname(__file__), "test_transform_config/test_config.json")
        test_config_based_transform = os.path.join(os.path.dirname(__file__), "test_legacy_exts/test_config_transform/")

        # apply config based transformation on model
        _ = convert_model(input_model=tf_net_path, transformations_config=config_path,
                          extensions=test_config_based_transform)

        # convert another model which would fail if custom transform from config_path applied
        tf_net_path = self.create_tf_model_no_sigmoid(temp_dir)
        _ = convert_model(input_model=tf_net_path, extensions=test_config_based_transform)

        # check that CustomReplacementRegistry.registry is cleared
        from openvino.tools.mo.front.common.custom_replacement_registry import CustomReplacementRegistry
        assert len(CustomReplacementRegistry.registry) == 0
