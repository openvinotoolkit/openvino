# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict


def wrap_ord_dict(func):
    """Wrap values in OrderedDict."""

    def wrapped(*args, **kwargs):
        items = func(*args, **kwargs)
        if isinstance(items, tuple):
            return OrderedDict([items])
        elif isinstance(items, list):
            return OrderedDict(items)
        elif isinstance(items, dict) or isinstance(items, OrderedDict):
            return items
        else:
            raise TypeError(
                "Decorated function '{}' returned '{}' but 'tuple', 'list', 'dict' or 'OrderedDict' expected"
                .format(func.__name__, type(items)))

    wrapped.unwrap = func
    return wrapped
