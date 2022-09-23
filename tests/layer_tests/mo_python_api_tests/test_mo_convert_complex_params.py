# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino.tools.mo.convert import InputCutInfo, LayoutMap

from common.mo_convert_test_class import CommonMOConvertTest
from common.tf_layer_test_class import save_to_pb
from openvino.runtime import Model, Layout, PartialShape, Shape, layout_helpers, Type, Dimension

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
                         'input':['Input1', 'Input2', 'Relu3']},
         'params_ref': {'input_shape': "[2,3,4],[2,3,4],[2,3,4]", 'input': 'Input1,Input2,Relu3'}},
        {'params_test': {'input_shape': [PartialShape([Dimension(), Dimension(1, 3), Dimension(4, -1), Dimension(-1, 5)]),
                                         [Dimension(), Dimension(1, 3), 4, Dimension(-1, 5)],
                                         [Dimension(), 3, Dimension(4, -1), Dimension(-1, 5)]],
                         'input':['Input1', 'Input2', 'Relu3']},
         'params_ref': {'input_shape': "[?,1..3,4..,..5],[?,1..3,4,..5],[?,3,4..,..5]", 'input': 'Input1,Input2,Relu3'}},
        {'params_test': {'input': [InputCutInfo("Relu1", Shape([3, 2]), Type(np.int32), None),
                                   InputCutInfo("Relu2", PartialShape([Dimension(3, 10), Dimension(2, -1)]), np.int32, None),
                                   InputCutInfo("Relu3", [3, 2], Type(np.int32), [1, 2, 3, 4, 5, 6])]},
         'params_ref': {'input': "Relu1[3 2]{i32},Relu2[3..10 2..]{i32},Relu3[3 2]{i32}->[1 2 3 4 5 6]"}},
        {'params_test': {'input': [("Relu1", Shape([3, 2]), Type(np.int32)),
                                   (np.int32, "Relu2", PartialShape([Dimension(3, 10), Dimension(2, -1)])),
                                   ([3, 2],"Relu3",  Type(np.int32))]},
         'params_ref': {'input': "Relu1[3 2]{i32},Relu2[3..10 2..]{i32},Relu3[3 2]{i32}"}},
        {'params_test': {'output': ["Sigmoid_0", "Sigmoid_2"]},
         'params_ref': {'output': "Sigmoid_0,Sigmoid_2"}},
        {'params_test': {'mean_values': {'Input1': [0.5,1.3,0.67], 'Input2':[4.2, 6.7, 3.15], 'Input3':[0.757, 4.6, 7.3]}},
         'params_ref': {'mean_values': "Input1[0.5,1.3,0.67],Input2[4.2,6.7,3.15],Input3[0.757,4.6,7.3]"}},
        {'params_test': {
            'mean_values': [[0.5, 1.3, 0.67], [4.2, 6.7, 3.15], [0.757, 4.6, 7.3]]},
         'params_ref': {'mean_values': "[0.5,1.3,0.67],[4.2,6.7,3.15],[0.757,4.6,7.3]"}},
        {'params_test': {'scale_values': {'Input1': [0.5,1.3,0.67], 'Input2':[4.2, 6.7, 3.15], 'Input3':[0.757, 4.6, 7.3]}},
         'params_ref': {'scale_values': "Input1[0.5,1.3,0.67],Input2[4.2,6.7,3.15],Input3[0.757,4.6,7.3]"}},
        {'params_test': {
            'scale_values': [[0.5, 1.3, 0.67], [4.2, 6.7, 3.15], [0.757, 4.6, 7.3]]},
         'params_ref': {'scale_values': "[0.5,1.3,0.67],[4.2,6.7,3.15],[0.757,4.6,7.3]"}},
        {'params_test': {
            'source_layout': {'Input1': Layout("nchw"), 'Input2': "nchw", 'Input3': "nc??"}},
         'params_ref': {'source_layout': "Input1(nchw),Input2(nchw),Input3(nc??)"}},
        {'params_test': {
            'target_layout': {'Input1': Layout("nhwc"), 'Input2': "nhwc", 'Input3': "n??c"}},
            'params_ref': {'target_layout': "Input1(nhwc),Input2(nhwc),Input3(n??c)"}},
        {'params_test': {
            'layout': {'Input1': LayoutMap(source_layout=Layout("nchw"), target_layout="nhwc"),
                       'Input2': LayoutMap(source_layout="nc??", target_layout=Layout("n??c")),
                       'Input3': LayoutMap(source_layout="abcd", target_layout="acdb")}},
            'params_ref': {'layout': "Input1(nchw->nhwc),Input2(nc??->n??c),Input3(abcd->acdb)"}},

    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mo_convert_tf_model(self, params, ie_device, precision, ir_version,
                                 temp_dir, use_new_frontend, use_old_api):
        tf_net_path = self.create_tf_model(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    test_data = [
        {'params_test': {'input_shape': PartialShape([2, 3, 4])},
         'params_ref': {'input_shape': "[2,3,4]"}},
        {'params_test': {'input_shape': [Dimension(), Dimension(1, 3), 4, Dimension(-1, 5)]},
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
        {'params_test': {'mean_values': [0.5, 1.3, 0.67]},
         'params_ref': {'mean_values': "[0.5,1.3,0.67]"}},
        {'params_test': {'scale_values': [0.5, 1.3, 0.67]},
         'params_ref': {'scale_values': "[0.5,1.3,0.67]"}},
        {'params_test': {'source_layout': Layout("nchw")},
         'params_ref': {'source_layout': "nchw"}},
        {'params_test': {'target_layout': Layout("nchw")},
         'params_ref': {'target_layout': "nchw"}},
        {'params_test': {'layout': LayoutMap(source_layout=Layout("nchw"), target_layout="nhwc")},
         'params_ref': {'layout': "nchw->nhwc"}},
        {'params_test': {'layout': Layout("nchw")},
         'params_ref': {'layout': "nchw"}}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_tf_model_single_input_output(self, params, ie_device, precision, ir_version,
                                                     temp_dir, use_new_frontend, use_old_api):
        tf_net_path = self.create_tf_model_single_input_output(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    test_data = [
        {
            'params_test': {'transform': ('MakeStateful', {'param_res_names': {'Input:0': 'Identity:0'}})},
            'params_ref': {'transform': "MakeStateful[param_res_names={\'Input:0\':\'Identity:0\'}]"}}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mo_convert_transform(self, params, ie_device, precision, ir_version,
                                  temp_dir, use_new_frontend, use_old_api):
        tf_net_path = self.create_tf_param_res_model(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)
