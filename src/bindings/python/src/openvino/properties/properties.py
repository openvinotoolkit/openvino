# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import BuiltinFunctionType
from typing import Callable, Any, Union

import openvino


class Property(str):
    def __new__(cls, prop: Callable[..., Any]):
        instance = super().__new__(cls, prop())
        instance.prop = prop
        return instance

    def __call__(self, *args: Any) -> Callable[..., Any]:
        if args is not None:
            return self.prop(*args)
        return self.prop()


def __append_property_to_module(content_, target_module_name):
    module = sys.modules[target_module_name]

    def base_getattr(name: str) -> None:
        raise AttributeError(
            f"Module '{module.__name__}' doesn't have the attribute with name '{name}'.")

    getattr_old = getattr(module, "__getattr__", base_getattr)

    def getattr_new(name: str) -> Union[Callable[..., Any], Any]:
        if content_.__name__ == name:
            return Property(content_)
        else:
            return getattr_old(name)

    module.__getattr__ = getattr_new # type: ignore


def __make_properties(target_module_name):
    for content in dir(openvino._pyopenvino.properties):
        content_ = getattr(openvino._pyopenvino.properties,content)
        if isinstance(content_, BuiltinFunctionType):
            __append_property_to_module(content_, target_module_name)
