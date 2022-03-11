# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Dimension, Function, PartialShape, Shape


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


def test_partial_shape():
    ps = PartialShape([1, 2, 3, 4])
    assert ps.is_static
    assert not ps.is_dynamic
    assert ps.rank == 4
    assert repr(ps) == "<PartialShape: {1,2,3,4}>"
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
    assert repr(shape) == "<Shape: {1, 2, 3}>"
    assert repr(ps) == "<PartialShape: {1,2,3}>"

    ps = PartialShape([Dimension(1), Dimension(2), Dimension(3), Dimension.dynamic()])
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.all_non_negative
    assert ps.rank == 4
    assert list(ps.get_min_shape()) == [1, 2, 3, 0]
    assert list(ps.get_max_shape())[3] > 1000000000
    assert repr(ps) == "<PartialShape: {1,2,3,?}>"
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
    assert repr(ps) == "<PartialShape: {1,2,3,?}>"

    ps = PartialShape.dynamic()
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == Dimension.dynamic()
    assert list(ps.get_min_shape()) == []
    assert list(ps.get_max_shape()) == []
    assert repr(ps) == "<PartialShape: ...>"

    ps = PartialShape.dynamic(r=Dimension(2))
    assert not ps.is_static
    assert ps.is_dynamic
    assert ps.rank == 2
    assert 2 == ps.rank
    assert list(ps.get_min_shape()) == [0, 0]
    assert list(ps.get_max_shape())[0] > 1000000000
    assert repr(ps) == "<PartialShape: {?,?}>"


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


def test_repr_dynamic_shape():
    shape = PartialShape([-1, 2])
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")
    model = parameter_a + parameter_b
    function = Function(model, [parameter_a, parameter_b], "simple_dyn_shapes_graph")

    assert repr(function) == "<Function: 'simple_dyn_shapes_graph' ({?,2})>"

    ops = function.get_ordered_ops()
    for op in ops:
        assert "{?,2}" in repr(op)


def test_discrete_type_info():
    data_shape = [6, 12, 10, 24]
    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)
    k = np.int32(3)
    axis = np.int32(1)
    n1 = ng.topk(data_parameter, k, axis, "max", "value")
    n2 = ng.topk(data_parameter, k, axis, "max", "value")
    n3 = ng.sin(0.2)

    assert n1.type_info.name == "TopK"
    assert n3.type_info.name == "Sin"
    assert n1.get_type_info().name == "TopK"
    assert n3.get_type_info().name == "Sin"
    assert n1.type_info.name == n2.type_info.name
    assert n1.type_info.version == n2.type_info.version
    assert n1.type_info.parent == n2.type_info.parent
    assert n1.get_type_info().name == n2.get_type_info().name
    assert n1.get_type_info().version == n2.get_type_info().version
    assert n1.get_type_info().parent == n2.get_type_info().parent
    assert n1.get_type_info().name != n3.get_type_info().name
    assert n1.get_type_info().name > n3.get_type_info().name
    assert n1.get_type_info().name >= n3.get_type_info().name
    assert n3.get_type_info().name < n1.get_type_info().name
    assert n3.get_type_info().name <= n1.get_type_info().name
