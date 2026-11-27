# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestScaledDotProductAttention(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (self.random.randn(1, 2, 8, 4, dtype=dtype),
                self.random.randn(1, 2, 16, 4, dtype=dtype),
                self.random.randn(1, 2, 16, 6, dtype=dtype))

    def create_model(self, mask, is_causal, dtype, mask_shape, enable_gqa):
        import torch.nn.functional as F
        import torch

        # Generate mask outside inner class to use test's seeded random
        attn_mask = None if not mask else torch.from_numpy(
            self.random.randint(0, 2, mask_shape, dtype=dtype))

        class aten_sdpa(torch.nn.Module):
            def __init__(self, mask=False, is_causal=False,
                         dtype=np.float32, mask_shape=None, enable_gqa=False) -> None:
                super().__init__()
                self.mask = attn_mask
                self.is_causal = False if (is_causal and mask) else is_causal
                self.enable_gqa = enable_gqa

            def forward(self, query, key, value):
                # torch export struggles with dynamic scale
                a = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   enable_gqa=self.enable_gqa)
                b = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   scale=torch.tensor(5, dtype=torch.float), enable_gqa=self.enable_gqa)
                return a, b

        return (aten_sdpa(mask, is_causal, dtype, mask_shape, enable_gqa),
                'aten::scaled_dot_product_attention')

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize('mask', (True, False))
    @pytest.mark.parametrize('is_causal', (True, False))
    @pytest.mark.parametrize('mask_shape', [(8, 16),
                                            (1, 16),
                                            (8, 1),
                                            (1, 1),
                                            (1, 8, 1),
                                            (1, 1, 8, 1),
                                            (1, 2, 8, 16)])
    @pytest.mark.parametrize('dyn_shapes', (True, False))
    @pytest.mark.parametrize('enable_gqa', (False, True))
    def test_scaled_dot_product_atten(self, ie_device, precision, ir_version,
                                      mask, is_causal, mask_shape, dyn_shapes, enable_gqa):
        if PytorchLayerTest.use_torch_export() and not mask and is_causal:
            pytest.xfail(reason="Unsupported case for torch.export")
        dtype = np.float32
        self._test(*self.create_model(mask, is_causal, dtype, mask_shape, enable_gqa),
                   ie_device, precision, ir_version, dynamic_shapes=dyn_shapes,
                   kwargs_to_prepare_input={"dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.parametrize(['mask', 'is_causal'], [(False, False),
                                                     (False, True),
                                                     (True, True),
                                                     (True, False)])
    @pytest.mark.parametrize('mask_shape', [(8, 16),
                                            (1, 16),
                                            (8, 1),
                                            (1, 1),
                                            (1, 8, 1),
                                            (1, 1, 8, 1),
                                            (1, 2, 8, 16)])
    @pytest.mark.parametrize('dyn_shapes', (True, False))
    def test_scaled_dot_product_atten_fp64(self, ie_device, precision,
                                           ir_version, mask, is_causal,
                                           mask_shape, dyn_shapes):
        if PytorchLayerTest.use_torch_export() and not mask and is_causal:
            pytest.xfail(reason="Unsupported case for torch.export")
        dtype = np.float64
        self._test(*self.create_model(mask, is_causal, dtype, mask_shape, enable_gqa=False),
                   ie_device, precision, ir_version, dynamic_shapes=dyn_shapes,
                   kwargs_to_prepare_input={"dtype": dtype})


class TestScaledDotProductAttentionWithGroupQuery(PytorchLayerTest):
    def _prepare_input(self, dtype):
        # with group size equal to 2 = 6 / 3
        return (self.random.randn(1, 7, 6, 8, 4, dtype=dtype),
                self.random.randn(1, 7, 3, 16, 4, dtype=dtype),
                self.random.randn(1, 7, 3, 16, 6, dtype=dtype))

    def create_model(self, mask, is_causal, dtype, mask_shape):
        import torch.nn.functional as F
        import torch

        # Generate mask outside inner class to use test's seeded random
        attn_mask = None if not mask else torch.from_numpy(
            self.random.randint(0, 2, mask_shape, dtype=dtype))

        class aten_sdpa(torch.nn.Module):
            def __init__(self, mask=False, is_causal=False,
                         dtype=np.float32, mask_shape=None) -> None:
                super().__init__()
                self.mask = attn_mask
                self.is_causal = False if (is_causal and mask) else is_causal

            def forward(self, query, key, value):
                # torch export struggles with dynamic scale
                a = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   enable_gqa=True)
                b = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   scale=torch.tensor(5, dtype=torch.float), enable_gqa=True)
                return a, b

        return (aten_sdpa(mask, is_causal, dtype, mask_shape),
                'aten::scaled_dot_product_attention')

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('mask', (True, False))
    @pytest.mark.parametrize('is_causal', (True, False))
    @pytest.mark.parametrize('mask_shape', [(8, 16),
                                            (1, 16),
                                            (8, 1),
                                            (1, 1),
                                            (1, 8, 1),
                                            (1, 1, 8, 1),
                                            (1, 1, 8, 16)])
    @pytest.mark.parametrize('dyn_shapes', (True, False))
    @pytest.mark.precommit_torch_export
    def test_scaled_dot_product_atten_with_gqa(self, ie_device, precision, ir_version,
                                               mask, is_causal, mask_shape, dyn_shapes):
        dtype = np.float32
        self._test(*self.create_model(mask, is_causal, dtype, mask_shape),
                   ie_device, precision, ir_version, dynamic_shapes=dyn_shapes,
                   kwargs_to_prepare_input={"dtype": dtype})


# Rank-4 [B, H, L, D] inputs with enable_gqa=True, the layout real LLMs use (Hq=8, L=5, S=16, D=4).
# name -> (Bq, Hq, Bkv, Hk, Hv, Dv)
GQA_L, GQA_S, GQA_D = 5, 16, 4
GQA_RANK4_LAYOUTS = {
    "gqa": (1, 8, 1, 2, 2, 4),             # group size 4
    "mqa": (1, 8, 1, 1, 1, 4),             # single K/V head
    "g1": (1, 8, 1, 8, 8, 4),              # Hq == Hkv: plain SDPA when the head counts are static
    "dk_ne_dv": (1, 8, 1, 2, 2, 6),        # value head size differs from query/key head size
    "kv_batch_bcast": (2, 8, 1, 2, 2, 4),  # K/V batch broadcast against Q batch
    "hk_ne_hv": (1, 8, 1, 2, 4, 4),        # key and value are repeated by different factors
}
# Layouts also run with torch.export, which ignores the conversion options below.
GQA_RANK4_EXPORT_LAYOUTS = ("gqa", "mqa", "g1")
# mask kind -> (attn_mask shape, dtype, passed as model input); "causal" means is_causal=True without a mask
# (torch rejects an explicit attn_mask together with is_causal=True).
GQA_RANK4_MASKS = {
    "none": None,
    "causal": None,
    "LS": ((GQA_L, GQA_S), np.float32, False),
    "11LS": ((1, 1, GQA_L, GQA_S), np.float32, False),
    "1HLS": ((1, 8, GQA_L, GQA_S), np.float32, False),
    "HLS": ((8, GQA_L, GQA_S), np.float32, False),
    "bool_1HLS": ((1, 8, GQA_L, GQA_S), bool, False),
    "input_1HLS": ((1, 8, GQA_L, GQA_S), np.float32, True),
}
# How the model reaches the translator, which decides what it can read from the input shapes:
#   script: TorchScript scripting, inputs of dynamic rank;
#   trace:  TorchScript tracing, static rank with dynamic dimensions;
#   static: convert_model with static input shapes, every dimension known (static group size).
GQA_CONVERSIONS = {
    "script": {"dynamic_shapes": True},
    "trace": {"dynamic_shapes": True, "trace_model": True},
    "static": {"dynamic_shapes": False, "use_convert_model": True},
}


class TestScaledDotProductAttentionGQARank4(PytorchLayerTest):
    def _prepare_input(self, layout, mask_kind):
        bq, hq, bkv, hk, hv, dv = GQA_RANK4_LAYOUTS[layout]
        q_len, kv_len, head_size = GQA_L, GQA_S, GQA_D
        inputs = (self.random.randn(bq, hq, q_len, head_size, dtype=np.float32),
                  self.random.randn(bkv, hk, kv_len, head_size, dtype=np.float32),
                  self.random.randn(bkv, hv, kv_len, dv, dtype=np.float32))
        mask = GQA_RANK4_MASKS[mask_kind]
        if mask is not None and mask[2]:
            inputs += (self.random.randn(*mask[0], dtype=np.float32),)
        return inputs

    def create_model(self, mask_kind):
        import torch.nn.functional as F

        mask = GQA_RANK4_MASKS[mask_kind]
        is_causal = mask_kind == "causal"
        attn_mask = None
        if mask is not None and not mask[2]:
            # Random values, so a mask applied to the wrong head or position changes the result.
            if mask[1] is bool:
                values = self.random.random_bool(*mask[0])
                values[..., 0] = True  # no fully masked row
            else:
                values = self.random.randn(*mask[0], dtype=np.float32)
            attn_mask = torch.from_numpy(values)

        class aten_sdpa_gqa(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mask = attn_mask
                self.is_causal = is_causal

            def forward(self, query, key, value):
                a = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask,
                                                   is_causal=self.is_causal, enable_gqa=True)
                b = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask,
                                                   is_causal=self.is_causal, scale=0.25, enable_gqa=True)
                return a, b

        class aten_sdpa_gqa_mask_input(torch.nn.Module):
            def forward(self, query, key, value, mask):
                a = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=True)
                b = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=0.25, enable_gqa=True)
                return a, b

        model = aten_sdpa_gqa_mask_input() if (mask is not None and mask[2]) else aten_sdpa_gqa()
        return model, "aten::scaled_dot_product_attention"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("layout", list(GQA_RANK4_LAYOUTS))
    @pytest.mark.parametrize("mask_kind", list(GQA_RANK4_MASKS))
    @pytest.mark.parametrize("conversion", list(GQA_CONVERSIONS))
    def test_scaled_dot_product_atten_gqa_rank4(self, ie_device, precision, ir_version, layout, mask_kind, conversion):
        if PytorchLayerTest.use_torch_export() and (conversion != "script" or layout not in GQA_RANK4_EXPORT_LAYOUTS):
            pytest.skip("torch.export ignores the TorchScript conversion options and runs a subset of layouts")
        self._test(*self.create_model(mask_kind), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"layout": layout, "mask_kind": mask_kind},
                   **GQA_CONVERSIONS[conversion])


class TestScaledDotProductAttentionGQARank3(PytorchLayerTest):
    # Unbatched [H, L, D] inputs: Hq=8 over Hkv=2.
    def _prepare_input(self):
        return (self.random.randn(8, GQA_L, GQA_D, dtype=np.float32),
                self.random.randn(2, GQA_S, GQA_D, dtype=np.float32),
                self.random.randn(2, GQA_S, GQA_D, dtype=np.float32))

    def create_model(self, is_causal):
        import torch.nn.functional as F

        class aten_sdpa_gqa_rank3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.is_causal = is_causal

            def forward(self, query, key, value):
                return F.scaled_dot_product_attention(query, key, value, is_causal=self.is_causal, enable_gqa=True)

        return aten_sdpa_gqa_rank3(), "aten::scaled_dot_product_attention"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("conversion", list(GQA_CONVERSIONS))
    def test_scaled_dot_product_atten_gqa_rank3(self, ie_device, precision, ir_version, is_causal, conversion):
        self._test(*self.create_model(is_causal), ie_device, precision, ir_version,
                   **GQA_CONVERSIONS[conversion])


class TestScaledDotProductAttentionGQAGraph:
    """Check that enable_gqa converts to an SDPA of the input rank with repeated key/value heads.

    Key and value heads are repeated by Unsqueeze -> Broadcast(BIDIRECTIONAL) -> Reshape, the form the SDPA
    fusions match.
    """

    class aten_sdpa_gqa(torch.nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)

    @staticmethod
    def _static_heads(ov_model):
        # Whether the translator saw static head counts for query, key and value.
        shapes = [param.get_partial_shape() for param in ov_model.get_parameters()]
        return all(shape.rank.is_static and shape[-3].is_static for shape in shapes)

    @staticmethod
    def _convert(conversion, example):
        # Returns the converted model and whether the translator saw static head counts.
        model = TestScaledDotProductAttentionGQAGraph.aten_sdpa_gqa()
        shapes = [ov.PartialShape(list(x.shape)) for x in example]
        if conversion == "trace_static":
            ov_model = ov.convert_model(model, example_input=example, input=shapes)
        elif conversion == "trace_dynamic":
            ov_model = ov.convert_model(model, example_input=example, input=[ov.PartialShape([-1] * 4)] * 3)
        elif conversion == "script_dynamic_rank":
            ov_model = ov.convert_model(torch.jit.script(model))
            static_heads = TestScaledDotProductAttentionGQAGraph._static_heads(ov_model)
            # Inputs are of dynamic rank here; set static shapes to make the graph checkable.
            ov_model.reshape(dict(zip(ov_model.inputs, shapes)))
            return ov_model, static_heads
        elif conversion == "export":
            ov_model = ov.convert_model(torch.export.export(model, example))
        else:
            raise ValueError(conversion)
        return ov_model, TestScaledDotProductAttentionGQAGraph._static_heads(ov_model)

    @staticmethod
    def _assert_repeated(ov_model, sdpa, port):
        reshape = sdpa.input_value(port).get_node()
        assert reshape.get_type_name() == "Reshape", f"SDPA input {port}: {reshape}"
        assert reshape.get_output_partial_shape(0).rank.get_length() == 4
        broadcast = reshape.input_value(0).get_node()
        assert broadcast.get_type_name() == "Broadcast", f"SDPA input {port}: {broadcast}"
        assert broadcast.get_attributes()["mode"] == "bidirectional", broadcast.get_attributes()
        unsqueeze = broadcast.input_value(0).get_node()
        assert unsqueeze.get_type_name() == "Unsqueeze", f"SDPA input {port}: {unsqueeze}"
        assert unsqueeze.input_value(1).get_node().get_type_name() == "Constant", "Unsqueeze axis must be a Constant"
        # [B, Hkv, 1, S, E]: the group axis follows the kv heads, so each kv head is repeated in place
        # (kv_head = q_head // G, as torch repeat_interleave), not tiled.
        unsq_shape = unsqueeze.get_output_partial_shape(0)
        assert unsq_shape.rank.get_length() == 5, unsq_shape
        assert unsq_shape[2] == ov.Dimension(1), unsq_shape
        # The repeat applies to its own input (key for port 1, value for port 2), directly.
        source = unsqueeze.input_value(0).get_node()
        assert source.get_friendly_name() == ov_model.get_parameters()[port].get_friendly_name(), source
        # GPU ExpandBroadcastReshapeSDPAFusion requires each node of the chain to have a single consumer.
        for node in (unsqueeze, broadcast, reshape):
            assert len(node.output(0).get_target_inputs()) == 1, node

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("layout", ["gqa", "mqa", "dk_ne_dv", "hk_ne_hv", "g1"])
    @pytest.mark.parametrize("conversion", ["trace_static", "trace_dynamic", "script_dynamic_rank", "export"])
    def test_scaled_dot_product_atten_gqa_graph(self, ie_device, precision, layout, conversion):
        if ie_device != "CPU" or precision != "FP32":
            pytest.skip("graph structure does not depend on device or precision")
        bq, hq, bkv, hk, hv, dv = GQA_RANK4_LAYOUTS[layout]
        example = (torch.randn(bq, hq, GQA_L, GQA_D),
                   torch.randn(bkv, hk, GQA_S, GQA_D),
                   torch.randn(bkv, hv, GQA_S, dv))
        ov_model, static_heads = self._convert(conversion, example)
        sdpas = [n for n in ov_model.get_ordered_ops() if n.get_type_name() == "ScaledDotProductAttention"]
        assert len(sdpas) == 1, [n.get_type_name() for n in ov_model.get_ordered_ops()]
        sdpa = sdpas[0]
        for port in range(sdpa.get_input_size()):
            rank = sdpa.get_input_partial_shape(port).rank
            assert rank.is_static, f"SDPA input {port} has dynamic rank"
            assert rank.get_length() <= 4, f"SDPA input {port} has rank {rank}"
        assert sdpa.get_input_partial_shape(0).rank.get_length() == 4
        if static_heads and hq == hk == hv:
            # Static group size 1: no repeat (an all-ones Broadcast would be removed by EliminateNopBroadcast
            # and the leftover Unsqueeze -> Reshape is not matched by the SDPA fusions).
            op_types = {n.get_type_name() for n in ov_model.get_ordered_ops()}
            assert "Broadcast" not in op_types, op_types
            for port in (1, 2):
                assert sdpa.input_value(port).get_node().get_type_name() == "Parameter"
        else:
            for port in (1, 2):
                self._assert_repeated(ov_model, sdpa, port)


class TestScaledDotProductAttentionGQAKVCacheFusion:
    """Check the converted enable_gqa attention over a stateful KV cache on CPU.

    The attention over the present key/value is converted from PyTorch; a KV cache (ReadValue -> Gather(beam_idx)
    -> Concat -> Assign) is then put in place of the present key/value inputs. StatefulSDPAFusion must turn the
    result into one ScaledDotProductAttention node that reads the un-repeated cache; the layouts that node can't
    execute must still compile. Either way the outputs match torch.
    """

    class aten_sdpa_gqa(torch.nn.Module):
        def __init__(self, scale=None):
            super().__init__()
            self.scale = scale

        def forward(self, query, present_key, present_value, attn_mask=None):
            return torch.nn.functional.scaled_dot_product_attention(query, present_key, present_value,
                                                                    attn_mask=attn_mask, scale=self.scale,
                                                                    enable_gqa=True)

    @staticmethod
    def _make_stateful_model(heads, head_size, prefix=(), scale=None, with_mask=False):
        import openvino.opset13 as ops
        from openvino import op

        num_q_heads, num_k_heads, num_v_heads = heads
        lead = [1, *prefix]
        example = [torch.randn(*lead, num_q_heads, 2, head_size),
                   torch.randn(*lead, num_k_heads, 5, head_size),
                   torch.randn(*lead, num_v_heads, 5, head_size)]
        shapes = [ov.PartialShape([-1, *prefix, h, -1, head_size]) for h in heads]
        if with_mask:
            example.append(torch.randn(1, 1, 2, 5))
            shapes.append(ov.PartialShape([1, 1, -1, -1]))
        attention = ov.convert_model(TestScaledDotProductAttentionGQAKVCacheFusion.aten_sdpa_gqa(scale),
                                     example_input=tuple(example), input=shapes)
        query, present_key, present_value, *mask = attention.get_parameters()
        query.output(0).get_tensor().set_names({"query"})
        if mask:
            mask[0].output(0).get_tensor().set_names({"attn_mask"})
        key = ops.parameter([-1, *prefix, num_k_heads, -1, head_size], np.float32, name="key")
        value = ops.parameter([-1, *prefix, num_v_heads, -1, head_size], np.float32, name="value")
        beam_idx = ops.parameter([-1], np.int32, name="beam_idx")
        batch = ops.gather(ops.shape_of(query), ops.constant([0]), ops.constant(0))
        assigns = []
        for present, current, num_heads, name in ((present_key, key, num_k_heads, "past_key"),
                                                  (present_value, value, num_v_heads, "past_value")):
            info = op.util.VariableInfo()
            info.data_shape = ov.PartialShape([-1, *prefix, num_heads, -1, head_size])
            info.data_type = ov.Type.f32
            info.variable_id = name
            cache_variable = op.util.Variable(info)
            empty_shape = ops.concat([batch, ops.constant([*prefix, num_heads, 0, head_size], dtype=np.int64)], 0)
            empty = ops.broadcast(ops.constant(np.float32(0)), empty_shape)
            past = ops.gather(ops.read_value(empty, cache_variable), beam_idx, ops.constant(0))
            cache = ops.concat([past, current], len(prefix) + 2)
            present.output(0).replace(cache.output(0))
            assigns.append(ops.assign(cache, cache_variable))
        return ov.Model(attention.get_results(), assigns, [query, key, value, beam_idx, *mask], "gqa_kv_cache")

    @staticmethod
    def _check_outputs(compiled, heads, head_size, prefix=(), scale=None, with_mask=False):
        # Prefill, then one decode step over the cache; every output must match torch.
        num_q_heads, num_k_heads, num_v_heads = heads
        request = compiled.create_infer_request()
        rng = np.random.RandomState(0)
        keys, values = [], []
        for length in (5, 1):
            inputs = {"query": rng.randn(1, *prefix, num_q_heads, length, head_size).astype(np.float32),
                      "key": rng.randn(1, *prefix, num_k_heads, length, head_size).astype(np.float32),
                      "value": rng.randn(1, *prefix, num_v_heads, length, head_size).astype(np.float32),
                      "beam_idx": np.zeros(1, dtype=np.int32)}
            keys.append(inputs["key"])
            values.append(inputs["value"])
            mask = None
            if with_mask:
                inputs["attn_mask"] = rng.randn(1, 1, length, sum(k.shape[-2] for k in keys)).astype(np.float32)
                mask = torch.from_numpy(inputs["attn_mask"])
            out = request.infer(inputs)[0]
            ref = torch.nn.functional.scaled_dot_product_attention(
                torch.from_numpy(inputs["query"]), torch.from_numpy(np.concatenate(keys, -2)),
                torch.from_numpy(np.concatenate(values, -2)), attn_mask=mask, scale=scale, enable_gqa=True).numpy()
            np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)

    @staticmethod
    def _compile(model, ie_device):
        compiled = ov.Core().compile_model(model, ie_device, {"INFERENCE_PRECISION_HINT": "f32",
                                                              "KV_CACHE_PRECISION": "f32"})
        op_types = [node.get_rt_info()["layerType"].astype(str) for node in compiled.get_runtime_model().get_ops()]
        return compiled, op_types

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("num_kv_heads,scale,with_mask", [(2, None, False), (1, None, False), (2, 0.125, True)])
    def test_scaled_dot_product_atten_gqa_kv_cache_fusion(self, ie_device, precision, num_kv_heads, scale, with_mask):
        if ie_device != "CPU" or precision != "FP32":
            pytest.skip("checks the CPU KV-cache fusion")
        heads, head_size = (8, num_kv_heads, num_kv_heads), 64
        model = self._make_stateful_model(heads, head_size, scale=scale, with_mask=with_mask)
        compiled, op_types = self._compile(model, ie_device)
        assert op_types.count("ScaledDotProductAttention") == 1, op_types
        assert not {"MatMul", "Softmax", "Subgraph"} & set(op_types), op_types
        self._check_outputs(compiled, heads, head_size, scale=scale, with_mask=with_mask)

    # Layouts the fused CPU node can't execute: they must compile and match torch.
    # name -> (heads (Hq, Hk, Hv), prefix dimensions, scale, with_mask)
    FALLBACK_CASES = {
        "rank5": ((8, 2, 2), (3,), None, False),
        "hk_ne_hv": ((8, 2, 4), (), None, False),
        "zero_scale": ((8, 2, 2), (), 0.0, True),
        "scale_without_mask": ((8, 2, 2), (), 0.125, False),
    }

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("case", list(FALLBACK_CASES))
    def test_scaled_dot_product_atten_gqa_kv_cache_fallback(self, ie_device, precision, case):
        if ie_device != "CPU" or precision != "FP32":
            pytest.skip("checks the CPU KV-cache fusion")
        heads, prefix, scale, with_mask = self.FALLBACK_CASES[case]
        head_size = 64
        model = self._make_stateful_model(heads, head_size, prefix, scale, with_mask)
        compiled, _ = self._compile(model, ie_device)
        self._check_outputs(compiled, heads, head_size, prefix, scale, with_mask)
