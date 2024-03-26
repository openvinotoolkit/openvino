# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Dimension, Symbol


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
