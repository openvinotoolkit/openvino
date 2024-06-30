# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from enum import Enum
from jax.lax import ConvDimensionNumbers


def enum_values_pass(value):
    if isinstance(value, Enum):
        return value.value
    return value


def conv_dimension_numbers_pass(value):
    if isinstance(value, ConvDimensionNumbers):
        return [
            list(value.lhs_spec),
            list(value.rhs_spec),
            list(value.out_spec)
        ]
    return value


def filter_element(value):
    passes = [enum_values_pass]
    for pass_ in passes:
        value = pass_(value)
    return value


def filter_ivalue(value):
    passes = [conv_dimension_numbers_pass]
    for pass_ in passes:
        value = pass_(value)
    return value
