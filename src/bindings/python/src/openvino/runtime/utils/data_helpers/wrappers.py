# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from collections.abc import Mapping
# TODO: remove this WA and refactor OVDict when Python3.8
# becomes minimal supported version.
try:
    from functools import singledispatchmethod
except ImportError:
    from singledispatchmethod import singledispatchmethod

from typing import Union

from openvino._pyopenvino import Tensor, ConstOutput
from openvino._pyopenvino import InferRequest as InferRequestBase


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))  # type: ignore


class _InferRequestWrapper(InferRequestBase):
    """InferRequest class with internal memory."""

    def __init__(self, other: InferRequestBase) -> None:
        # Private memeber to store newly created shared memory data
        self._inputs_data = None
        super().__init__(other)


class OVDict(Mapping):
    """Custom OpenVINO dictionary with inference results.

    This class is a dict-like object. It provides possibility to
    address each element with three key types:

    * `openvino.runtime.ConstOutput` - port of the output
    * `int` - index of the output
    * `str` - names of the output

    This class follows `frozenset`/`tuple` concept of immutability.
    It is prohibited to assign new items or edit them.

    To revert to the previous behavior use `to_dict` method which
    return shallow copy of underlaying dictionary.
    Note: It removes addressing feature! New dictionary keeps
          only `ConstOutput` keys.

    If a tuple return value is needed, use `to_tuple` method which
    converts values to the tuple.

    :Example:

    .. code-block:: python
        # Reverts to the previous behavior of the native dict
        result = dict(request.infer(...))
        # ... or alternatively:
        result = request.infer(...).to_dict()

    .. code-block:: python
        # To dispatch outputs of multi-ouput inference:
        out1, out2, out3, _ = request.infer(...).values()
        # ... or alternatively:
        out1, out2, out3, _ = request.infer(...).to_tuple()
    """
    def __init__(self, d) -> None:
        self._data = d

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self):
        return self._data.__repr__()

    def __get_key(self, index: int) -> ConstOutput:
        return list(self._data.keys())[index]

    @singledispatchmethod
    def __getitem_impl(self, key: Union[ConstOutput, int, str]):
        raise TypeError("Unknown key type!")

    @__getitem_impl.register
    def _(self, key: ConstOutput) -> np.ndarray:
        return self._data[key]

    @__getitem_impl.register
    def _(self, key: int) -> np.ndarray:
        return self._data[self.__get_key(key)]

    @__getitem_impl.register
    def _(self, key: str) -> np.ndarray:
        return self._data[self.__get_key(self.names().index(key))]

    def __getitem__(self, key):
        return self.__getitem_impl(key)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def names(self) -> list:
        """Return a name of every output key."""
        return [key.get_any_name() for key in self._data.keys()]

    def to_dict(self) -> dict:
        """Convert to a native dictionary.

        Function performs shallow copy, thus any modifications to
        original values may affect this class as well.
        """
        return self._data

    def to_tuple(self) -> tuple:
        """Convert values of this dictionary to a tuple."""
        return tuple(self._data.values())
