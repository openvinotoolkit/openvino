# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

test_data_3D = [
    dict(input_shape=[1, 50, 50], output_shapes=[[1, 50, 25], [1, 50, 25]], axis=2),
    dict(input_shape=[2, 50, 50], output_shapes=[[2, 20, 50], [2, 15, 50], [2, 15, 50]], axis=1),
    dict(input_shape=[4, 50, 50],
         output_shapes=[[1, 50, 50], [1, 50, 50], [1, 50, 50], [1, 50, 50]], axis=0)]

test_data_4D = [
    dict(input_shape=[1, 32, 800, 800], output_shapes=[[1, 16, 800, 800], [1, 16, 800, 800]],
         axis=1),
    dict(input_shape=[4, 32, 80, 80],
         output_shapes=[[4, 8, 80, 80], [4, 8, 80, 80], [4, 8, 80, 80], [4, 8, 80, 80]],
         axis=1),
    dict(input_shape=[2, 21, 80, 80],
         output_shapes=[[2, 7, 80, 80], [2, 7, 80, 80], [2, 7, 80, 80]], axis=1),
    dict(input_shape=[3, 21, 80, 80],
         output_shapes=[[3, 14, 80, 80], [3, 5, 80, 80], [3, 2, 80, 80]], axis=1),
    dict(input_shape=[3, 21, 80, 80],
         output_shapes=[[1, 21, 80, 80], [1, 21, 80, 80], [1, 21, 80, 80]], axis=0),
    dict(input_shape=[3, 21, 80, 80],
         output_shapes=[[3, 21, 20, 80], [3, 21, 35, 80], [3, 21, 25, 80]], axis=2),
    dict(input_shape=[3, 21, 80, 80],
         output_shapes=[[3, 21, 80, 40], [3, 21, 80, 10], [3, 21, 80, 30]], axis=3)]

test_data_5D = [
    dict(input_shape=[1, 50, 50, 80, 60],
         output_shapes=[[1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60]], axis=2),
    dict(input_shape=[1, 50, 50, 80, 60], output_shapes=[[1, 25, 50, 80, 60], [1, 25, 50, 80, 60]],
         axis=1)]

test_multiple_out = [
    dict(input_shape=[3, 10, 10],
         output_shapes=[[1, 10, 10],
                        [1, 10, 10],
                        [1, 10, 10]],
         axis=0,
         output_names=['h', 'b', 'l']),
    dict(input_shape=[1, 50, 50, 80, 60],
         output_shapes=[[1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60]],
         axis=2,
         output_names=['k', 'p', 'a', 'r', 's']),
    dict(input_shape=[1, 4, 3],
         output_shapes=[[1, 1, 3],
                        [1, 1, 3],
                        [1, 1, 3],
                        [1, 1, 3],
                        [1, 1, 3]],
         axis=1,
         output_names=['inp4', 'inp1', 'inp3', 'inp2'])
]

test_multiple_out_with_add = [
    dict(input_shape=[3, 10, 10],
         output_shapes=[[1, 10, 10],
                        [1, 10, 10],
                        [1, 10, 10]],
         axis=0,
         output_names=['h', 'b', 'l', 'c', 'p']
         ),
    dict(input_shape=[1, 50, 50, 80, 60],
         output_shapes=[[1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60],
                        [1, 50, 10, 80, 60]],
         axis=2,
         output_names=['k', 'p', 'a', 'r', 's', 'l', 'w']),
    dict(input_shape=[1, 4, 3],
         output_shapes=[[1, 1, 3],
                        [1, 1, 3],
                        [1, 1, 3],
                        [1, 1, 3],
                        [1, 1, 3]],
         axis=1,
         output_names=['inp4', 'inp1', 'inp5', 'inp2', 'inp3', 'inp33'])
]

test_multiple_out_with_identity = [
    dict(input_shape=[3, 10, 10],
         output_shapes=[[1, 10, 10],
                        [1, 10, 10],
                        [1, 10, 10]],
         axis=0,
         split_out_names=['h', 'b', 'l'],
         identity_names=['i1', 'i2', 'i3'],
         output_names=['h', 'b', 'l', 'i3'],
         ),
]


class TestSplitConcat(OnnxRuntimeLayerTest):
    # TODO Add test with default values (axis=0)
    def create_split_concat_net(self, input_shape, output_shapes, axis, ir_version):
        """
            ONNX net                             IR net

            Input->Split->Concat->Output   =>    Input->Split->Concat

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        outputs, split = [], []
        for id, output_shape in enumerate(output_shapes):
            helper.make_tensor_value_info('output_{}'.format(id), TensorProto.FLOAT, output_shape)
            outputs.append('output_{}'.format(id))
            split.append(output_shape[axis])

        # Output for concat
        output_concat = helper.make_tensor_value_info('output_concat', TensorProto.FLOAT,
                                                      input_shape)

        node_split_def = onnx.helper.make_node(
            'Split',
            inputs=['input'],
            outputs=outputs,
            axis=axis,
            split=split
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=outputs,
            outputs=['output_concat'],
            axis=axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_split_def, node_concat_def],
            'test_split_model',
            [input],
            [output_concat],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model')

        #
        #   Create reference IR net
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    # TODO Add test with default values (axis=0)
    def create_split_concat_net_const(self, input_shape, output_shapes, axis, ir_version):
        """
            ONNX net                                               IR net

            Input(const)->Split->Concat--->Concat->Output   =>    Input--->Concat
                                  Input-'                         Const-'

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto
        import numpy as np

        concat_axis = 0
        concat_output_shape = input_shape.copy()
        concat_output_shape[concat_axis] *= 2

        const_number = np.prod(input_shape)
        constant = np.random.randint(-127, 127, const_number).astype(float)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        outputs, split = [], []
        for id, output_shape in enumerate(output_shapes):
            helper.make_tensor_value_info('output_{}'.format(id), TensorProto.FLOAT, output_shape)
            outputs.append('output_{}'.format(id))
            split.append(output_shape[axis])

        # Output for concat
        output_concat = helper.make_tensor_value_info('output_dyn_concat', TensorProto.FLOAT,
                                                      concat_output_shape)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=input_shape,
                vals=constant,
            ),
        )

        node_split_def = onnx.helper.make_node(
            'Split',
            inputs=['const1'],
            outputs=outputs,
            axis=axis,
            split=split
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=outputs,
            outputs=['output_concat'],
            axis=axis
        )

        node_dyn_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'output_concat'],
            outputs=['output_dyn_concat'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_split_def, node_concat_def, node_dyn_concat_def],
            'test_split_model',
            [input],
            [output_concat],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model')

        #
        #   Create reference IR net
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_concat_net(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_concat_net(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_concat_net(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_3D_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_split_concat_net_const(**params, ir_version=ir_version), ie_device,
            precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_4D_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_split_concat_net_const(**params, ir_version=ir_version), ie_device,
            precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_5D_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_split_concat_net_const(**params, ir_version=ir_version), ie_device,
            precision, ir_version, temp_dir=temp_dir)


class TestSplit(OnnxRuntimeLayerTest):
    # TODO Add test with default values (axis=0)
    def create_split_net(self, input_shape, output_shapes, axis, ir_version):
        """
            ONNX net                       IR net

            Input->Split->Output   =>    Input->Split

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        outputs, split = [], []
        for id, output_shape in enumerate(output_shapes):
            out = helper.make_tensor_value_info('output_{}'.format(id), TensorProto.FLOAT,
                                                output_shape)
            outputs.append((out, 'output_{}'.format(id)))
            split.append(output_shape[axis])

        node_split_def = onnx.helper.make_node(
            'Split',
            inputs=['input'],
            outputs=['node_{}'.format(x[1]) for x in outputs],
            axis=axis,
            split=split
        )
        nodes = [node_split_def]

        for x in outputs:
            nodes.append(onnx.helper.make_node(
                'Elu',
                inputs=['node_{}'.format(x[1])],
                outputs=[x[1]]
            ))

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_split_model',
            [input],
            [x[0] for x in outputs],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model')

        #
        #   Create reference IR net
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    def create_split_net_ordered_outputs(self, input_shape, output_shapes, axis, output_names,
                                         ir_version):
        """
            ONNX net                             IR net

            Input->Split->Output1   =>    Input->Split->Output1
                        ->Output2   =>                ->Output2
                        ->Output3   =>                ->Output3

        """
        #
        #   Create ONNX model
        #
        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = input_shape

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)

        output_list = []
        for i, output_name in enumerate(output_names):
            output_list.append(
                helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shapes[i]))

        node = onnx.helper.make_node('Split', inputs=['input'], outputs=output_names, axis=axis)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node],
            'split_model',
            [input],
            output_list,
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model_outputs_order')

        ref_net = None

        return onnx_net, ref_net

    def create_split_net_ordered_outputs_with_add(self, input_shape, output_shapes, axis,
                                                  output_names, ir_version):
        """
        This test checks the case when graph has a node that is connected with Result and some other operation
        from single output port.

            ONNX net                                           IR net

                 Input                                          Input
                  |                                               |
                Split                                           Split
              |       |           ... |                         |   |     ....   |
            Ouput1    Output2      OutputN                     |    |           Result_N
              \      /                                        /\   / \
               Add                                          /   Add   \
                                                        Result_0 |     Result_1
                                                                Result_N+1

        """
        #
        #   Create ONNX model
        #
        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = input_shape

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)

        add_output_name1 = output_names[len(output_names) - 2]
        add_output_name2 = output_names[len(output_names) - 1]
        outputs_without_add = output_names[:len(output_names) - 2]

        output_list = []
        for i, output_name in enumerate(outputs_without_add):
            output_list.append(
                helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shapes[i]))

        node = onnx.helper.make_node('Split', inputs=['input'], outputs=outputs_without_add,
                                     axis=axis)
        node_add1 = helper.make_node(
            'Add',
            inputs=[outputs_without_add[1], outputs_without_add[2]],
            outputs=[add_output_name1]
        )
        node_add2 = helper.make_node(
            'Add',
            inputs=[add_output_name1, outputs_without_add[2]],
            outputs=[add_output_name2]
        )

        output_list = output_list + [
            helper.make_tensor_value_info(add_output_name1, TensorProto.FLOAT,
                                          output_shapes[0])] + [
                          helper.make_tensor_value_info(add_output_name2, TensorProto.FLOAT,
                                                        output_shapes[0])]

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node, node_add1, node_add2],
            'split_model',
            [input],
            output_list,
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model_outputs_order')

        ref_net = None

        return onnx_net, ref_net

    def create_split_net_ordered_outputs_multiple_tensor_names(self, input_shape, output_shapes,
                                                               axis, split_out_names,
                                                               identity_names, output_names,
                                                               ir_version):
        """
        This test checks the case of multiple tensor names on connection incoming to Result. In this case
        Result name is equal to one of tensor names from the list.

            ONNX net                             IR net

            Input->Split->Identity1->Identity2->Identity3 -> Output1
                        ->Output2
                        ->Output3


            IR net

            Input->Split->Result1 - this connection has tensor names from Split, Identity1, Identity2, Identity3 ops
                        ->Result2
                        ->Result3

        """
        #
        #   Create ONNX model
        #
        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = input_shape

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)

        output_list = []
        for i, output_name in enumerate(split_out_names):
            output_list.append(
                helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shapes[i]))
        output_list.append(
            helper.make_tensor_value_info(identity_names[2], TensorProto.FLOAT, output_shapes[i]))

        node = onnx.helper.make_node('Split', inputs=['input'], outputs=split_out_names, axis=axis)
        identity1 = onnx.helper.make_node('Identity', inputs=[split_out_names[0]],
                                          outputs=[identity_names[0]])
        identity2 = onnx.helper.make_node('Identity', inputs=[identity_names[0]],
                                          outputs=[identity_names[1]])
        identity3 = onnx.helper.make_node('Identity', inputs=[identity_names[1]],
                                          outputs=[identity_names[2]])

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node, identity1, identity2, identity3],
            'split_model',
            [input],
            output_list,
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model_outputs_order')

        ref_net = None

        return onnx_net, ref_net

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_split_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_multiple_out)
    def test_split_outputs_order(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_split_net_ordered_outputs(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   output_names=params['output_names'])

    @pytest.mark.parametrize("params", test_multiple_out_with_add)
    def test_split_outputs_order_multiple_connection_before_result_case(self, params, ie_device,
                                                                        precision, ir_version,
                                                                        temp_dir):
        self._test(*self.create_split_net_ordered_outputs_with_add(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   output_names=params['output_names'])

    @pytest.mark.parametrize("params", test_multiple_out_with_identity)
    def test_split_outputs_order_multiple_tensors_before_result_case(self,
                                                                     params,
                                                                     ie_device,
                                                                     precision,
                                                                     ir_version,
                                                                     temp_dir):
        self._test(*self.create_split_net_ordered_outputs_multiple_tensor_names(**params,
                                                                                ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   output_names=params['output_names'])
