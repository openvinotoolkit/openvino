# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

import torch
import torch.nn as nn
from common.layer_test_class import CommonLayerTest, check_ir_version
from unit_tests.utils.graph import build_graph


class PytorchLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        path = os.path.join(save_path, 'model.onnx')
        self.torch_model = framework_model
        torch.onnx.export(self.torch_model, self.var, path, output_names=['output'])
        assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(save_path)
        return path

    def get_framework_results(self, inputs_dict, model_path):
        return {'output': self.torch_model(*self.var).detach().numpy()}


class EmbeddingBagModel(torch.nn.Module):
    def __init__(self, n, m, indices_shape=None, per_sample_weights=False, mode="sum"):
        super(EmbeddingBagModel, self).__init__()
        EE = nn.EmbeddingBag(n, m, mode=mode, sparse=True)
        self.W = np.random.uniform(low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)).astype(
            np.float32)
        EE.weight.data = torch.tensor(self.W, requires_grad=True)
        self.embedding_bag = EE
        if per_sample_weights:
            self.per_sample_weights = torch.randn(indices_shape)
        else:
            self.per_sample_weights = None


class TestPytorchEmbeddingBag(PytorchLayerTest):
    def _prepare_input(self, inputs_dict):
        assert 'input' in inputs_dict and 'offsets' in inputs_dict, "input and offsets should be in inputs_dict"
        indices, offsets = self.var
        inputs_dict['input'] = indices.numpy().astype(np.int32)
        inputs_dict['offsets'] = offsets.numpy().astype(np.int32)
        return inputs_dict

    def create_net(self, n, m, emb_batch_size, ir_version, per_sample_weights=False, offsets=None):
        """
            Pytorch net                         IR net

            Input->EmbeddingBag->Output   =>    Input->Gather/SparseWeightedSum

        """
        #   Create Pytorch model
        EE = EmbeddingBagModel(n, m, indices_shape=[emb_batch_size],
                               per_sample_weights=per_sample_weights)

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input_weights_data': {'kind': 'data', 'value': EE.W.flatten()},
                'weights': {'kind': 'op', 'type': 'Const'},
                'weights_data': {'shape': EE.W.shape, 'kind': 'data'},
                'indices': {'kind': 'op', 'type': 'Parameter'},
                'indices_data': {'kind': 'data'},
                'node': {'kind': 'op', 'type': 'EmbeddingBagOffsetsSum'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            edges = [('input_weights_data', 'weights'),
                     ('weights', 'weights_data'),
                     ('indices', 'indices_data'),
                     ('weights_data', 'node'),
                     ('node', 'node_data'),
                     ('node_data', 'result')
                     ]
            if offsets is not None:
                nodes_attributes.update({
                    'offsets': {'kind': 'op', 'type': 'Parameter'},
                    'offsets_data': {'kind': 'data'},
                    'node_data': {'shape': [len(offsets), m], 'kind': 'data'},
                })
                edges.extend([
                    ('offsets', 'offsets_data'),
                    ('indices_data', 'node'),
                    ('offsets_data', 'node'),
                ])
            else:
                nodes_attributes.update({
                    'input_shape_data': {'kind': 'data', 'value': [-1]},
                    'shape': {'kind': 'op', 'type': 'Const'},
                    'shape_data': {'shape': [1], 'kind': 'data'},
                    'reshape': {'kind': 'op', 'type': 'Reshape'},
                    'reshape_data': {'shape': [emb_batch_size], 'kind': 'data'},
                    'input_offsets_data': {'kind': 'data', 'value': np.arange(0, 128, 2)},
                    'offsets': {'kind': 'op', 'type': 'Const'},
                    'offsets_data': {'shape': [int(emb_batch_size / 2)], 'kind': 'data'},
                    'node_data': {'shape': [int(emb_batch_size / 2), m], 'kind': 'data'},
                })
                edges.extend([
                    ('input_shape_data', 'shape'),
                    ('shape', 'shape_data'),
                    ('indices_data', 'reshape'),
                    ('shape_data', 'reshape'),
                    ('reshape', 'reshape_data'),
                    ('reshape_data', 'node'),
                    ('input_offsets_data', 'offsets'),
                    ('offsets', 'offsets_data'),
                    ('offsets_data', 'node'),
                ])

            ref_net = build_graph(nodes_attributes, edges)
        if offsets is not None:
            self.var = (torch.from_numpy(np.random.choice(n, emb_batch_size)).long(),
                        torch.from_numpy(np.array(offsets)).long())
        else:
            self.var = (
                torch.from_numpy(
                    np.random.choice(n, emb_batch_size).reshape(int(emb_batch_size / 2),
                                                                2)).long(),)
        return EE, ref_net

    test_data = [
        dict(n=1460, m=16, emb_batch_size=128),
        dict(n=1460, m=16, emb_batch_size=128, offsets=np.arange(0, 128)),
        dict(n=1460, m=16, emb_batch_size=128, offsets=[0, 2, 6, 20, 80]),
        # dict(n=1460, m=16, emb_batch_size=128, offsets=[0, 2, 6, 20, 80], per_sample_weights=True),
        # per_sample_weights not supported in ONNX
        dict(n=1460, m=16, emb_batch_size=128, offsets=[0, 2, 6, 20, 20, 80])  # empty bag case
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_pytorch_embedding_bag(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params), ie_device, precision, ir_version, temp_dir=temp_dir)
