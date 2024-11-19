# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino._pyopenvino.util import numpy_to_c, replace_node, replace_output_update_name
from openvino.package_utils import get_cmake_path
from functools import wraps
from typing import Callable, Any


def deprecated(name: Any = None, version: str = "", message: str = "", stacklevel: int = 2) -> Callable[..., Any]:
    """Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}" and runs the function.

    :param version: The version in which the code will be removed.
    :param message: A message explaining why the function is deprecated and/or what to use instead.
    """

    def decorator(wrapped: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(wrapped)
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            # it must be imported here; otherwise, there are errors with no loaded DLL for Windows
            from openvino._pyopenvino.util import deprecation_warning

            deprecation_warning(wrapped.__name__ if name is None else name, version, message, stacklevel)
            return wrapped(*args, **kwargs)

        return wrapper

    return decorator


# WA method since Python 3.11 does not support @classmethod and @property chain,
# currently only read-only properties are supported.
class _ClassPropertyDescriptor(object):
    def __init__(self, fget: Callable):
        self.fget = fget

    def __get__(self, obj: Any, cls: Any = None) -> Any:
        if cls is None:
            cls = type(obj)
        return self.fget.__get__(obj, cls)()


def classproperty(func: Any) -> _ClassPropertyDescriptor:
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


def deprecatedclassproperty(name: Any = None, version: str = "", message: str = "", stacklevel: int = 2) -> Callable[[Any], _ClassPropertyDescriptor]:
    def decorator(wrapped: Any) -> _ClassPropertyDescriptor:
        func = classproperty(wrapped)

        # Override specific instance
        def _patch(instance: _ClassPropertyDescriptor, func: Callable[..., Any]) -> None:
            cls_: Any = type(instance)

            class _(cls_):  # noqa: N801
                @func
                def __get__(self, obj: Any, cls: Any = None) -> Any:
                    return super().__get__(obj, cls)

            instance.__class__ = _

        # Add `deprecated` decorator on the top of `__get__`
        _patch(func, deprecated(name, version, message, stacklevel))
        return func
    return decorator
