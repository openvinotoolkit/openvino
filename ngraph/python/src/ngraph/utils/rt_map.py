# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Overrides pybind PyRTMap class methods."""

from typing import Any

import ngraph.pyngraph

from ngraph.pyngraph import Variant
from ngraph.impl import VariantInt, VariantString
from ngraph.exceptions import UserInputError


def _convert_to_variant(item: Any) -> Variant:
    """Convert value to Variant class, otherwise throw error."""
    if isinstance(item, Variant):
        return item
    variant_mapping = {
        int: VariantInt,
        str: VariantString,
    }

    new_type = variant_mapping.get(type(item), None)

    if new_type is None:
        raise UserInputError("Cannot map value to any of registered Variant classes", str(item))

    return new_type(item)


binding_copy = ngraph.pyngraph.PyRTMap.__setitem__


def _setitem(self: ngraph.pyngraph.PyRTMap, arg0: str, arg1: Any) -> None:
    """Override setting values in dictionary."""
    binding_copy(self, arg0, _convert_to_variant(arg1))


ngraph.pyngraph.PyRTMap.__setitem__ = _setitem
