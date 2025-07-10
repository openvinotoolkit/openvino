# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest

from openvino import Dimension, Model, PartialShape, Shape

import openvino.opset13 as ov


def test_dimension():
    dim = Dimension()
    assert dim.is_dynamic
    assert not dim.is_static
    assert repr(dim) == "<Dimension: ?>"

    dim = Dimension.dynamic()
    assert dim.is_dynamic
    assert not dim.is_static
    assert repr(dim) == "<Dimension: ?>"

    dim = Dimension(10)
    assert dim.is_static
    assert len(dim) == 10
    assert dim.get_length() == 10
    assert dim.get_min_length() == 10
    assert dim.get_max_length() == 10
    assert repr(dim) == "<Dimension: 10>"

    dim = Dimension(5, 15)
    assert dim.is_dynamic
    assert dim.get_min_length() == 5
    assert dim.get_max_length() == 15
    assert repr(dim) == "<Dimension: 5..15>"


def test_dimension_comparisons():
    d1 = Dimension.dynamic()
    d2 = Dimension.dynamic()
    assert d1 == d2
    assert d1 == -1
    assert d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert d2.relaxes(d1)
    assert d2.compatible(d1)
    assert d2.same_scheme(d1)

    d1 = Dimension.dynamic()
    d2 = Dimension(3)
    assert d1 != d2
    assert d2 == 3
    assert not d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert not d2.relaxes(d1)
    assert d2.compatible(d1)
    assert not d2.same_scheme(d1)

    d1 = Dimension(3)
    d2 = Dimension(3)
    assert d1 == d2
    assert d1.refines(d2)
    assert d1.relaxes(d2)
    assert d2.refines(d1)
    assert d2.relaxes(d1)
    assert d2.compatible(d1)
    assert d2.same_scheme(d1)

    d1 = Dimension(4)
    d2 = Dimension(3)
    assert d1 != d2
    assert not d1.refines(d2)
    assert not d1.relaxes(d2)
    assert not d2.refines(d1)
    assert not d2.relaxes(d1)
    assert not d2.compatible(d1)
    assert not d2.same_scheme(d1)

    dim = Dimension("?")
    assert dim == Dimension()

    dim = Dimension("1")
    assert dim == Dimension(1)

    dim = Dimension("..10")
    assert dim == Dimension(-1, 10)

    dim = Dimension("10..")
    assert dim == Dimension(10, -1)

    dim = Dimension("5..10")
    assert dim == Dimension(5, 10)

    with pytest.raises(RuntimeError) as e:
        dim = Dimension("C")
    assert 'Cannot parse dimension: "C"' in str(e.value)

    with pytest.raises(RuntimeError) as e:
        dim = Dimension("?..5")
    assert 'Cannot parse min bound: "?"' in str(e.value)

    with pytest.raises(RuntimeError) as e:
        dim = Dimension("5..?")
    assert 'Cannot parse max bound: "?"' in str(e.value)


def test_partial_shape():
    ps = PartialShape([1, 2, 3, 4])
    assert ps.is_static
    assert not ps.is_dynamic
    assert ps.rank == 4
    assert repr(ps) == "<PartialShape: [1,2,3,4]>"
    assert ps.get_dimension(0) == Dimension(1)
    assert ps.get_dimension(1) == Dimension(2)
    assert ps.get_dimension(2) == Dimension(3)
    assert ps.get_dimension(3) == Dimension(4)

    shape = Shape([1, 2, 3])
    ps = PartialShape(shape)
    assert ps.is_static
    assert not ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 3
    assert list(ps.get_shape()) == [1, 2, 3]
    assert list(ps.get_max_shape()) == [1, 2, 3]
    assert list(ps.get_min_shape()) == [1, 2, 3]
    assert list(ps.to_shape()) == [1, 2, 3]
    assert repr(shape) == "<Shape: [1,2,3]>"
    assert repr(ps) == "<PartialShape: [1,2,3]>"

    ps = PartialShape([Dimension(1), Dimension(2), Dimension(3), Dimension.dynamic()])
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 4
    assert list(ps.get_min_shape()) == [1, 2, 3, 0]
    assert list(ps.get_max_shape())[3] > 1000000000
    assert repr(ps) == "<PartialShape: [1,2,3,?]>"
    assert ps.get_dimension(0) == Dimension(1)
    assert ps.get_dimension(1) == Dimension(2)
    assert ps.get_dimension(2) == Dimension(3)
    assert ps.get_dimension(3) == Dimension.dynamic()

    ps = PartialShape([1, 2, 3, -1])
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 4
    assert list(ps.get_min_shape()) == [1, 2, 3, 0]
    assert list(ps.get_max_shape())[3] > 1000000000
    assert repr(ps) == "<PartialShape: [1,2,3,?]>"

    ps = PartialShape.dynamic()
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == Dimension.dynamic()
    assert list(ps.get_min_shape()) == []
    assert list(ps.get_max_shape()) == []
    assert repr(ps) == "<PartialShape: [...]>"

    ps = PartialShape.dynamic(rank=Dimension(2))
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == 2
    assert 2 == ps.rank
    assert list(ps.get_min_shape()) == [0, 0]
    assert list(ps.get_max_shape())[0] > 1000000000
    assert repr(ps) == "<PartialShape: [?,?]>"

    shape_list = [(1, 10), [2, 5], 4, Dimension(2), "..10"]
    ref_ps = PartialShape(
        [
            Dimension(1, 10),
            Dimension(2, 5),
            Dimension(4),
            Dimension(2),
            Dimension(-1, 10),
        ],
    )
    assert PartialShape(shape_list) == ref_ps
    assert PartialShape(tuple(shape_list)) == ref_ps

    with pytest.raises(TypeError) as e:
        PartialShape([(1, 2, 3)])
    assert (
        "Two elements are expected in tuple(lower, upper) "
        "for dynamic dimension, but 3 elements were given." in str(e.value)
    )

    with pytest.raises(TypeError) as e:
        PartialShape([("?", "?")])
    assert (
        "Incorrect pair of types (<class 'str'>, <class 'str'>) "
        "for dynamic dimension, ints are expected." in str(e.value)
    )

    with pytest.raises(TypeError) as e:
        PartialShape([range(10)])
    assert (
        "Incorrect type <class 'range'> for dimension. Expected types are: "
        "int, str, openvino.Dimension, list/tuple with lower "
        "and upper values for dynamic dimension." in str(e.value)
    )

    ps = PartialShape("[...]")
    assert ps == PartialShape.dynamic()

    ps = PartialShape("[?, 3, ..224, 28..224]")
    assert ps == PartialShape([Dimension(-1), Dimension(3), Dimension(-1, 224), Dimension(28, 224)])

    with pytest.raises(RuntimeError) as e:
        ps = PartialShape("[?,,3]")
    assert 'Cannot get vector of dimensions! "[?,,3]" is incorrect' in str(e.value)

    shape = Shape()
    assert len(shape) == 0

    shape = PartialShape("[?, 3, ..224, 28..224, 25..]")
    copied_shape = copy.copy(shape)
    assert shape == copied_shape, "Copied shape {0} is not equal to original shape {1}.".format(copied_shape, shape)

    shape = PartialShape("[...]")
    copied_shape = copy.copy(shape)
    assert shape == copied_shape, "Copied shape {0} is not equal to original shape {1}.".format(copied_shape, shape)

    shape = PartialShape([Dimension(-1, 100), 25, -1])
    copied_shape = copy.copy(shape)
    assert shape == copied_shape, "Copied shape {0} is not equal to original shape {1}.".format(copied_shape, shape)

    shape = PartialShape("[?, 3, ..224, 28..224, 25..]")
    copied_shape = copy.deepcopy(shape)
    assert shape == copied_shape, "Copied shape {0} is not equal to original shape {1}.".format(copied_shape, shape)

    shape = PartialShape("[...]")
    copied_shape = copy.deepcopy(shape)
    assert shape == copied_shape, "Copied shape {0} is not equal to original shape {1}.".format(copied_shape, shape)

    shape = PartialShape([Dimension(-1, 100), 25, -1])
    copied_shape = copy.deepcopy(shape)
    assert shape == copied_shape, "Copied shape {0} is not equal to original shape {1}.".format(copied_shape, shape)

    ps = PartialShape.dynamic(rank=3)
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == 3
    assert list(ps.get_min_shape()) == [0, 0, 0]
    assert repr(ps) == "<PartialShape: [?,?,?]>"


def test_partial_shape_compatible():
    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1.compatible(ps2)

    ps1 = PartialShape([3])
    ps2 = PartialShape.dynamic()
    assert ps1.compatible(ps2)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape([4])
    assert ps1.compatible(ps2)

    ps1 = PartialShape([2, -1, 3, -1, 5])
    ps2 = PartialShape([2, -1, -1, 4, 5])
    assert ps1.compatible(ps2)

    ps1 = PartialShape([2, -1, 3, -1, 5])
    ps2 = PartialShape([1, -1, -1, 4, 5])
    assert not ps1.compatible(ps2)


def test_partial_shape_same_scheme():
    ps1 = PartialShape([1, 2, -1])
    ps2 = PartialShape([1, 3, -1])
    assert not ps1.same_scheme(ps2)

    ps1 = PartialShape([1, 2, -1])
    ps2 = PartialShape([1, 2, -1])
    assert ps1.same_scheme(ps2)

    ps1 = PartialShape([1, 2, 3])
    ps2 = PartialShape([1, 2, 3])
    assert ps1.same_scheme(ps2)

    ps1 = PartialShape([-1, 2, 3])
    ps2 = PartialShape([1, -1, 3])
    assert not ps1.same_scheme(ps2)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1.same_scheme(ps2)


def test_partial_shape_refinement():
    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1.refines(ps2)
    assert ps1.relaxes(ps2)
    assert ps2.refines(ps1)
    assert ps2.relaxes(ps1)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape([3, -1, 7, 9])
    assert not ps1.refines(ps2)
    assert ps1.relaxes(ps2)
    assert ps2.refines(ps1)
    assert not ps2.relaxes(ps1)

    ps1 = PartialShape.dynamic()
    ps2 = PartialShape([3, 5, 7, 9])
    assert not ps1.refines(ps2)
    assert ps1.relaxes(ps2)
    assert ps2.refines(ps1)
    assert not ps2.relaxes(ps1)


@pytest.mark.parametrize("shape_to_compare", [[1, 2, 3], (1, 2, 3)])
def test_shape_equals(shape_to_compare):
    shape = Shape([1, 2, 3])
    assert shape == shape_to_compare


def test_partial_shape_equals():
    ps1 = PartialShape.dynamic()
    ps2 = PartialShape.dynamic()
    assert ps1 == ps2

    ps1 = PartialShape([1, 2, 3])
    ps2 = PartialShape([1, 2, 3])
    assert ps1 == ps2

    shape = Shape([1, 2, 3])
    ps = PartialShape([1, 2, 3])
    assert shape == ps
    assert shape == ps.to_shape()

    ps1 = PartialShape.dynamic(rank=3)
    ps2 = PartialShape.dynamic(rank=3)
    assert ps1 == ps2

    ps = PartialShape([1, 2, 3])
    tuple_ps = (1, 2, 3)
    list_ps = [1, 2, 3]
    assert ps == tuple_ps
    assert ps == list_ps

    ps = PartialShape.dynamic(rank=3)
    tuple_ps = (0, 0, 0)
    list_ps = [0, 0, 0]
    assert ps.get_min_shape() == tuple_ps
    assert ps.get_min_shape() == list_ps

    ps = PartialShape.dynamic()
    tuple_ps = ()
    list_ps = []
    assert ps.get_min_shape() == tuple_ps
    assert ps.get_min_shape() == list_ps

    ps = PartialShape([Dimension(1), Dimension(2), Dimension(3), Dimension.dynamic()])
    tuple_ps = (1, 2, 3, 0)
    list_ps = [1, 2, 3, 0]
    assert ps.get_min_shape() == tuple_ps
    assert ps.get_min_shape() == list_ps

    ps = PartialShape([Dimension(1, 10), Dimension(2), Dimension(3)])
    tuple_ps_min = (1, 2, 3)
    tuple_ps_max = (10, 2, 3)
    list_ps_min = [1, 2, 3]
    list_ps_max = [10, 2, 3]
    assert ps.get_min_shape() == tuple_ps_min
    assert ps.get_max_shape() == tuple_ps_max
    assert ps.get_min_shape() == list_ps_min
    assert ps.get_max_shape() == list_ps_max

    with pytest.raises(TypeError) as e:
        ps = PartialShape.dynamic()
        tuple_ps = ()
        assert ps == tuple_ps
    assert (
        "Cannot compare dynamic shape with <class 'tuple'>" in str(e.value)
    )

    with pytest.raises(TypeError) as e:
        ps = PartialShape.dynamic()
        list_ps = []
        assert ps == list_ps
    assert (
        "Cannot compare dynamic shape with <class 'list'>" in str(e.value)
    )


def test_input_shape_read_only():
    shape = Shape([1, 10])
    param = ov.parameter(shape, dtype=np.float32)
    model = Model(ov.relu(param), [param])
    ref_shape = model.input().shape
    ref_shape[0] = Dimension(3)
    assert model.input().shape == shape


def test_repr_dynamic_shape():
    shape = PartialShape([-1, 2])
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    param_sum = parameter_a + parameter_b
    # set tensor name to have deterministic output name of model (default use unique node name)
    param_sum.output(0).set_names({"sum"})
    model = Model(param_sum, [parameter_a, parameter_b], "simple_dyn_shapes_graph")

    assert (
        repr(model)
        == "<Model: 'simple_dyn_shapes_graph'\ninputs["
        + "\n<ConstOutput: names[A] shape[?,2] type: f32>,"
        + "\n<ConstOutput: names[B] shape[?,2] type: f32>\n]"
        + "\noutputs[\n<ConstOutput: names[sum] shape[?,2] type: f32>\n]>"
    )

    ops = model.get_ordered_ops()
    for op in ops:
        assert "[?,2]" in repr(op)


def test_discrete_type_info():
    data_shape = [6, 12, 10, 24]
    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)
    k_val = np.int32(3)
    axis = np.int32(1)
    n1 = ov.topk(data_parameter, k_val, axis, "max", "value")
    n2 = ov.topk(data_parameter, k_val, axis, "max", "value")
    n3 = ov.sin(0.2)

    assert n1.type_info.name == "TopK"
    assert n3.type_info.name == "Sin"
    assert n1.get_type_info().name == "TopK"
    assert n3.get_type_info().name == "Sin"
    assert n1.type_info.name == n2.type_info.name
    assert n1.type_info.version_id == n2.type_info.version_id
    assert n1.type_info.parent == n2.type_info.parent
    assert n1.get_type_info().name == n2.get_type_info().name
    assert n1.get_type_info().version_id == n2.get_type_info().version_id
    assert n1.get_type_info().parent == n2.get_type_info().parent
    assert n1.get_type_info().name != n3.get_type_info().name
    assert n1.get_type_info().name > n3.get_type_info().name
    assert n1.get_type_info().name >= n3.get_type_info().name
    assert n3.get_type_info().name < n1.get_type_info().name
    assert n3.get_type_info().name <= n1.get_type_info().name


@pytest.mark.parametrize("shape_type", [Shape, PartialShape])
def test_shape_negative_index(shape_type):
    shape = shape_type([1, 2, 3, 4, 5])
    assert shape[-1] == 5
    assert shape[-3] == 3
    assert shape[-5] == 1


@pytest.mark.parametrize("shape_type", [Shape, PartialShape])
def test_shape_slicing_step(shape_type):
    shape = shape_type([1, 2, 3, 4, 5])
    assert list(shape[0:2]) == [1, 2]
    assert list(shape[0:3:2]) == [1, 3]
    assert list(shape[::2]) == [1, 3, 5]
    assert list(shape[1::2]) == [2, 4]
    assert list(shape[::-1]) == [5, 4, 3, 2, 1]
    assert list(shape[::-2]) == [5, 3, 1]
