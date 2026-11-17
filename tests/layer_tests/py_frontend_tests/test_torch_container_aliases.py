# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Tuple

import numpy as np
import openvino as ov
import pytest
import torch


class ContainerMetadataMutation(torch.nn.Module):
    __constants__ = ["mutation", "right", "update_other"]

    def __init__(self, mutation, right, update_other):
        super().__init__()
        self.mutation = mutation
        self.right = right
        self.update_other = update_other

    def forward(self, cache: Tuple[torch.Tensor, torch.Tensor]):
        return self.mutate(cache)

    def mutate(self, cache: Tuple[torch.Tensor, torch.Tensor]):
        left, right = cache
        member = right if self.right else left
        other = left if self.right else right
        view = member[:, :2]
        if self.update_other:
            other.add_(1)
        if self.mutation == "chain":
            member.transpose_(0, 1).unsqueeze_(0)
        elif self.mutation == "transpose":
            member.transpose_(0, 1)
        elif self.mutation == "unsqueeze":
            member.unsqueeze_(0)
        else:
            member.squeeze_(0)
        if self.update_other:
            other.unsqueeze_(-1)
            member.add_(2)
        return cache, view


class NestedContainerMetadataMutation(ContainerMetadataMutation):
    def forward(self, cache: Tuple[Tuple[torch.Tensor, torch.Tensor]]):
        inner, = cache
        _, view = self.mutate(inner)
        return cache, view


class ListMetadataMutation(torch.nn.Module):
    __constants__ = ["operation"]

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, cache: List[torch.Tensor]):
        left, right = cache
        view = left[:, :2]
        if self.operation == "storage":
            cache[0].add_(1)
        cache[0].transpose_(0, 1)
        if self.operation == "metadata":
            cache[0].unsqueeze_(0)
        return cache, left, view


@pytest.mark.precommit
@pytest.mark.nightly
def test_torch_metadata_mutation_after_unrelated_append():
    class Model(torch.nn.Module):
        def forward(self, cache: List[torch.Tensor], extra: torch.Tensor, target: torch.Tensor):
            left, right = cache
            before = left + right
            cache.append(extra)
            target.unsqueeze_(0)
            return before, torch.cat(cache), target

    model = torch.jit.script(Model())
    inputs = ([torch.ones(2, 3), torch.full((2, 3), 2.0)], torch.full((2, 3), 3.0), torch.arange(6.0).reshape(2, 3))
    converted = ov.convert_model(model, example_input=deepcopy(inputs))
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    results = list(compiled([value.numpy() for value in (*inputs[0], *inputs[1:])]).values())
    assert len(results) == 3
    for actual, reference in zip(results, model(*deepcopy(inputs))):
        np.testing.assert_array_equal(actual, reference.numpy())


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("mutate_clone", [False, True])
@pytest.mark.parametrize("operation", ["clone", "detach"])
def test_torch_tensor_metadata_identity(mutate_clone, operation):
    class Model(torch.nn.Module):
        __constants__ = ["mutate_clone", "operation"]

        def __init__(self):
            super().__init__()
            self.mutate_clone = mutate_clone
            self.operation = operation

        def forward(self, source):
            clone = source.clone() if self.operation == "clone" else source.detach()
            target = clone if self.mutate_clone else source
            target.unsqueeze_(0)
            return source, clone

    model = torch.jit.script(Model())
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    converted = ov.convert_model(model, example_input=source)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    results = list(compiled([source.numpy()]).values())
    expected = model(source.clone())
    assert len(results) == 2
    for actual, reference in zip(results, expected):
        np.testing.assert_array_equal(actual, reference.numpy())


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("operation", ["transpose", "storage", "metadata"])
def test_torch_list_member_metadata_preserves_views(operation):
    inputs = [torch.arange(6, dtype=torch.float32).reshape(2, 3), torch.ones(2, 3)]
    model = torch.jit.script(ListMetadataMutation(operation))
    converted = ov.convert_model(model, example_input=(deepcopy(inputs),))
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    cache, left, view = model(deepcopy(inputs))
    results = list(compiled([x.numpy() for x in inputs]).values())
    assert len(results) == 4
    for expected, actual in zip((*cache, left, view), results):
        np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.precommit
@pytest.mark.nightly
def test_torch_wrapped_chunk_base_mutation():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    graph = torch._C.parse_ir("""graph(%base : Tensor):
      %two : int = prim::Constant[value=2]()
      %zero : int = prim::Constant[value=0]()
      %one : int = prim::Constant[value=1]()
      %parts : Tensor[] = aten::chunk(%base, %two, %zero)
      %wrapped : (Tensor[]) = prim::TupleConstruct(%parts)
      %alias : Tensor[] = prim::TupleUnpack(%wrapped)
      %a : Tensor, %b : Tensor = prim::ListUnpack(%alias)
      %updated : Tensor = aten::add_(%base, %one, %one)
      return (%a, %b)
    """)
    frontend = ov.frontend.FrontEndManager().load_by_framework("pytorch")
    decoder = TorchScriptPythonDecoder(None, graph, alias_db=graph.alias_db())
    input_model = frontend.load(decoder)
    input_model.set_element_type(input_model.get_inputs()[0], ov.Type.f32)
    input_model.set_partial_shape(input_model.get_inputs()[0], ov.PartialShape([4, 3]))
    converted = frontend.convert(input_model)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    results = list(compiled([source]).values())
    assert len(results) == 2
    for actual, expected in zip(results, np.split(source + 1, 2)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("mutation,right,update_other,nested", [
    ("transpose", False, False, False),
    ("unsqueeze", False, False, False),
    ("squeeze", True, False, True),
    ("chain", True, False, False),
    ("chain", False, True, True),
    ("transpose", True, True, False),
])
def test_torch_container_metadata_mutation(mutation, right, update_other, nested):
    generator = torch.Generator().manual_seed(42)
    rows = 1 if mutation == "squeeze" else 2
    inputs = tuple(torch.randn(rows, 3, generator=generator) for _ in range(2))
    inputs = (inputs,) if nested else inputs
    module = NestedContainerMetadataMutation if nested else ContainerMetadataMutation
    model = torch.jit.script(module(mutation, right, update_other))
    converted = ov.convert_model(model, example_input=(deepcopy(inputs),))
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for columns in (3, 5):
        inputs = tuple(torch.randn(rows, columns, generator=generator) for _ in range(2))
        cache, view = model(deepcopy((inputs,) if nested else inputs))
        cache = cache[0] if nested else cache
        actual = list(compiled([member.numpy() for member in inputs]).values())
        assert len(actual) == 3
        for reference, result in zip((*cache, view), actual):
            np.testing.assert_array_equal(result, reference.numpy())


class ContainerMutation(torch.nn.Module):
    def __init__(self, nested, mutation):
        super().__init__()
        self.nested = nested
        self.mutation = mutation

    def forward(self, source, cache):
        left, right = cache[0] if self.nested else cache
        before = left + right
        left_view = left[:, :, :2]
        right_view = right[:, :, :2]
        if self.mutation == "view":
            left_view.copy_(source[:, :, :2])
        elif self.mutation == "right":
            right.copy_(source)
        else:
            left.copy_(source)
            if self.mutation == "repeated":
                left.add_(2)
            elif self.mutation == "siblings":
                right.copy_(source + 5)
        values = cache[0] if self.nested else cache
        return before, left, right, left_view, right_view, values[0], values[1]


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("container,nested,mutation", [
    (list, False, "copy"), (tuple, False, "right"),
    (list, False, "view"), (tuple, True, "view"),
    (tuple, True, "repeated"), (list, True, "siblings"),
])
def test_torch_container_input_mutation(container, nested, mutation):
    generator = torch.Generator().manual_seed(42)

    def make_inputs(batch):
        source = torch.randn(batch, 4, 3, generator=generator)
        elements = container(torch.randn(batch, 4, 3, generator=generator) for _ in range(2))
        cache = container([elements]) if nested else elements
        return source, cache

    inputs = make_inputs(1)
    model = torch.jit.trace(ContainerMutation(nested, mutation), deepcopy(inputs),
                           strict=False, check_trace=False)
    converted = ov.convert_model(model, example_input=deepcopy(inputs))
    assert len(converted.inputs) == 3
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        source, cache = make_inputs(batch)
        expected = model(*deepcopy((source, cache)))
        elements = cache[0] if nested else cache
        actual = list(compiled([source.numpy()] + [element.numpy() for element in elements]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)


class ChunkCacheMutation(torch.nn.Module):
    def forward(self, source, cache):
        b, c, x = source.chunk(3, dim=1)
        cache[0].copy_(b * x)
        return c * cache[0], cache


class SequenceViewMutation(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, source, data):
        base = data.clone()
        if self.operation == "split":
            parts = torch.split_with_sizes(base, [2, 2], dim=1)
        else:
            parts = torch.unbind(base.reshape(base.shape[0], 2, 2, 3), dim=1)
        before = parts[0] + parts[1]
        parts[0].copy_(source)
        parts[1].add_(2)
        return base, before, parts[0], parts[1]


class SequenceBaseMutation(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, data):
        base = data.clone()
        if self.operation == "chunk":
            left, right = base.chunk(2, dim=1)
        else:
            left, right = base.split([2, 2], dim=1)
        before = left + right
        base.add_(1)
        after = left + right
        right.add_(2)
        return base, before, after, left, right


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("operation", ["chunk", "split"])
def test_torch_sequence_base_mutation(operation):
    generator = torch.Generator().manual_seed(42)
    data = torch.randn(2, 4, 3, generator=generator)
    model = torch.jit.trace(SequenceBaseMutation(operation), data)
    converted = ov.convert_model(model, example_input=data)
    converted.reshape({converted.input(0): ov.PartialShape([-1, 4, 3])})
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        data = torch.randn(batch, 4, 3, generator=generator)
        expected = model(data)
        actual = list(compiled([data.numpy()]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("operation,capture", [("split", "torchscript"), ("unbind", "export")])
def test_torch_sequence_view_mutation(operation, capture):
    generator = torch.Generator().manual_seed(42)
    inputs = (torch.randn(2, 2, 3, generator=generator), torch.randn(2, 4, 3, generator=generator))
    model = SequenceViewMutation(operation)
    if capture == "torchscript":
        model = torch.jit.trace(model, deepcopy(inputs))
        converted = ov.convert_model(model, example_input=deepcopy(inputs))
    else:
        batch = torch.export.Dim("batch", min=1)
        exported = torch.export.export(model, deepcopy(inputs), dynamic_shapes=({0: batch}, {0: batch}))
        converted = ov.convert_model(exported)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        inputs = (torch.randn(batch, 2, 3, generator=generator), torch.randn(batch, 4, 3, generator=generator))
        expected = model(*deepcopy(inputs))
        actual = list(compiled([element.numpy() for element in inputs]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.precommit
@pytest.mark.nightly
def test_torch_chunk_cache_mutation():
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(1, 12, 3, generator=generator)
    cache = [torch.randn(1, 4, 3, generator=generator) for _ in range(2)]
    inputs = source, cache
    model = torch.jit.trace(ChunkCacheMutation(), deepcopy(inputs), strict=False, check_trace=False)
    converted = ov.convert_model(model, example_input=deepcopy(inputs))
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        source = torch.randn(batch, 12, 3, generator=generator)
        cache = [torch.randn(batch, 4, 3, generator=generator) for _ in range(2)]
        output, updated_cache = model(*deepcopy((source, cache)))
        expected = [output] + updated_cache
        actual = list(compiled([source.numpy()] + [element.numpy() for element in cache]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("sequence_input,mapping,error,metadata", [
    (0, [1, 0], None, False), (1, [1, 0], None, False), (1, [1, 0], None, True),
    (0, [0], "preserve the number of elements", False),
    (0, [0, 1, 0], "preserve the number of elements", False),
    (0, [0, 0], "exactly one element", False), (0, [0, 2], "out of range", False),
])
def test_torch_decoder_sequence_output_order(sequence_input, mapping, error, metadata):
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    ir = """graph(%left : Tensor, %right : Tensor, %source : Tensor):
      %pair : (Tensor, Tensor) = prim::TupleConstruct(%left, %right)
      %first : Tensor, %second : Tensor = prim::TupleUnpack(%pair)
      %blocking : bool = prim::Constant[value=0]()
      %updated : Tensor = aten::copy_(%first, %source, %blocking)
      return (%left, %right, %second)
    """
    if metadata:
        ir = ir.replace("%blocking : bool", "%blocking : int").replace(
            "aten::copy_(%first, %source, %blocking)", "aten::unsqueeze_(%first, %blocking)")
    graph = torch._C.parse_ir(ir)

    class ReversedSequenceDecoder(TorchScriptPythonDecoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sequence_queries = 0
            if self.get_op_type() == "custom::reversed_sequence" and sequence_input == 1:
                self.raw_inputs.insert(0, list(graph.inputs())[0])
                self._input_ids.insert(0, self.raw_inputs[0].unique())

        def get_op_type(self):
            if isinstance(self.graph_element, torch.Node) and self.graph_element.kind() == "prim::TupleUnpack":
                return "custom::reversed_sequence"
            return super().get_op_type()

        def get_input_sequences(self):
            self.sequence_queries += 1
            return {sequence_input: mapping} if self.get_op_type() == "custom::reversed_sequence" else {}

        def visit_subgraph(self, visitor):
            for node in self.graph_element.nodes():
                decoder = ReversedSequenceDecoder(None, node, alias_db=self.alias_db)
                self.m_decoders.append(decoder)
                visitor(decoder)

    decoder = ReversedSequenceDecoder(None, graph, alias_db=graph.alias_db())
    frontend = ov.frontend.FrontEndManager().load_by_framework("pytorch")
    frontend.add_extension(ov.frontend.ConversionExtension(
        "custom::reversed_sequence",
        lambda context: list(reversed(context.get_input(sequence_input).get_node().input_values()))))
    input_model = frontend.load(decoder)
    for place in input_model.get_inputs():
        input_model.set_element_type(place, ov.Type.f32)
    if error:
        with pytest.raises(RuntimeError, match=error):
            frontend.convert(input_model)
        return
    converted = frontend.convert(input_model)
    assert all(child.sequence_queries == bool(child.inputs()) for child in decoder.m_decoders)
    converted.reshape({port: ov.PartialShape([-1, 4, 3]) for port in converted.inputs})
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    generator = np.random.default_rng(42)
    for batch in (1, 2):
        left, right, source = [generator.standard_normal((batch, 4, 3)).astype(np.float32) for _ in range(3)]
        actual = list(compiled([left, right, source]).values())
        assert len(actual) == 3
        expected = (left, right[None], left) if metadata else (left, source, left)
        for result, expected in zip(actual, expected):
            np.testing.assert_allclose(result, expected)
