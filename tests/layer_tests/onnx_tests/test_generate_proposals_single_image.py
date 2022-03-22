# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import Caffe2OnnxLayerTest


class TestGenerateProposals(Caffe2OnnxLayerTest):
    skip_framework = False

    def _prepare_input(self, inputs_dict):
        N = 1
        A = 6
        H = 10
        W = 8

        inputs_dict["scores"] = np.random.rand(1, A, H, W).astype('float32')
        inputs_dict["bbox_deltas"] = np.random.rand(1, A * 4, H, W).astype('float32')
        inputs_dict["im_info"] = np.array([[1000, 1000, 1]]).astype('float32')
        inputs_dict["anchors"] = np.reshape(np.arange(A * 4),
                                            [A, 4]).astype('float32')

        #for input in inputs_dict.keys():
            #inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_generate_proposals(self):
        """
            ONNX net

            Input->generate_proposals->Output   =>   Only accuracy check

        """

        N = 1
        A = 6
        H = 10
        W = 8

        #   Create ONNX model
        import onnx
        from onnx import helper
        from onnx import TensorProto

        # Input for generate proposals
        scores_shape = [N, A, H, W]
        bbox_deltas_shape = [N, 4 * A, H, W]
        im_info_shape = [N, 3]
        anchors_shape = [A, 4]

        scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, scores_shape)
        bbox_deltas = helper.make_tensor_value_info('bbox_deltas', TensorProto.FLOAT, bbox_deltas_shape)
        im_info = helper.make_tensor_value_info('im_info', TensorProto.FLOAT, im_info_shape)
        anchors = helper.make_tensor_value_info('anchors', TensorProto.FLOAT, anchors_shape)

        # Output for generate proposals
        rois_shape = [-1, 4]
        rois_probs_shape = [-1]
        rois = helper.make_tensor_value_info('rois', TensorProto.FLOAT, rois_shape)
        rois_probs = helper.make_tensor_value_info('rois_probs', TensorProto.FLOAT, rois_probs_shape)

        generate_proposals = onnx.helper.make_node(
            'GenerateProposals',
            inputs=['scores', 'bbox_deltas', 'im_info', 'anchors'],
            outputs=['rois', 'rois_probs'],
            pre_nms_topN=100,
            post_nms_topN=60,
            nms_thresh=0.7,
            min_size=3,
            spatial_scale=1.0/16,
        )
            #legacy_plus_one=True
            #angle_bound_on=,
            #angle_bound_lo=,
            #angle_bound_hi=,
            #clip_angle_thresh=,

        inputs = ['scores', 'bbox_deltas', 'im_info', 'anchors']

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [generate_proposals],
            'test_generate_proposals',
            [scores, bbox_deltas, im_info, anchors],
            [rois, rois_probs],
        )

        # Create the model (ModelProto)
        onnx_net = helper.make_model(graph_def, producer_name='test_generate_proposals_model')

        # We do not create reference graph, as it's too complicated to construct it
        # Moreover, IR reader do not support TensorIterator layers
        # So we return None to skip IR comparision

        return onnx_net, None

    @pytest.mark.precommit
    @pytest.mark.timeout(3600)
    def test_generate_proposals_simple_precommit(self, ie_device, precision, ir_version,
                                   temp_dir, api_2):
        self._test(*self.create_generate_proposals(), ie_device, precision, ir_version,
                   temp_dir=temp_dir, infer_timeout=3600, api_2=api_2)

    @pytest.mark.nightly
    def test_generate_proposals_nightly(self, ie_device, precision, ir_version, temp_dir,
                          api_2):
        self._test(*self.create_generate_proposals(), ie_device, precision, ir_version,
                   temp_dir=temp_dir, api_2=api_2)

if __name__ == '__main__':
    test_generate_proposal = TestGenerateProposals()
    test_generate_proposal.test_generate_proposals_simple_precommit(ie_device='CPU', precision='FP32', ir_version=11, temp_dir='/tmp/pytest/', api_2=True)
