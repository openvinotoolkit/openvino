# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Dimension, Symbol


def test_dynamic_dimension():
    dim = Dimension()
    assert dim.is_dynamic
    assert str(dim) == "?"
    assert dim.to_string() == "?"
    assert str(dim.__repr__) == "<bound method PyCapsule.__repr__ of <Dimension: ?>>"


def test_dynamic_dimension_with_bounds():
    dim = Dimension(2, 5)
    assert str(dim) == "2..5"
    assert dim.to_string() == "2..5"
    assert not dim.is_static
    assert dim.is_dynamic
    assert dim.get_min_length() == 2
    assert dim.min_length == 2
    assert dim.get_max_length() == 5
    assert dim.max_length == 5


def test_static_dimension():
    dim = Dimension(2)
    assert str(dim) == "2"
    assert dim.to_string() == "2"
    assert dim.is_static
    assert not dim.is_dynamic
    assert len(dim) == 2
    assert dim.get_length() == 2


def test_dim_same_scheme():
    assert Dimension().same_scheme(Dimension()) is True
    assert Dimension(3).same_scheme(Dimension(3)) is True
    assert Dimension(3).same_scheme(Dimension(4)) is False
    assert Dimension().same_scheme(Dimension(4)) is False


def test_dim_compatible():
    assert Dimension().compatible(Dimension()) is True
    assert Dimension(3).compatible(Dimension(3)) is True
    assert Dimension(3).compatible(Dimension(4)) is False
    assert Dimension().compatible(Dimension(4)) is True


def test_dim_relax():
    assert Dimension().relaxes(Dimension()) is True
    assert Dimension(3).relaxes(Dimension(3)) is True
    assert Dimension(3).relaxes(Dimension(4)) is False
    assert Dimension().relaxes(Dimension(4)) is True


def test_dim_refine():
    assert Dimension().refines(Dimension()) is True
    assert Dimension(3).refines(Dimension(3)) is True
    assert Dimension(3).refines(Dimension(4)) is False
    assert Dimension().refines(Dimension(4)) is False


def test_symbol():
    dimension = Dimension()
    assert not dimension.has_symbol(), "Check: Default created Dimension has no symbol: Dimension.has_symbol()"
    assert not dimension.get_symbol(), "Check: Default created Dimension symbol is null: Symbol.__bool__"

    symbol = Symbol()
    dimension.set_symbol(symbol)
    assert dimension.has_symbol(), "Check: After setting the symbol, Dimension has symbol: Dimension.has_symbol()"
    assert dimension.get_symbol(), "Check: After setting the symbol, Dimension symbol isn't null: Symbol.__bool__"

    new_dimension = Dimension()
    assert dimension.get_symbol() != new_dimension.get_symbol(), "Check: Two symbols are not equal: Symbol.__eq__"

    new_dimension.set_symbol(dimension.get_symbol())
    assert dimension.get_symbol() == new_dimension.get_symbol(), "Check: Two symbols are equal: Symbol.__eq__"


def test_symbol_hash():
    symbol = Symbol()
    assert isinstance(hash(symbol), int)

    hash1 = hash(symbol)
    hash2 = hash(symbol)
    assert hash1 == hash2

    symbol1 = Symbol()
    symbol2 = Symbol()
    assert hash(symbol1) != hash(symbol2)

    symbols = {symbol1: "symbol1", symbol2: "symbol2"}
    assert symbols[symbol1] == "symbol1"
