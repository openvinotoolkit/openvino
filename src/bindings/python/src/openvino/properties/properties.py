# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.utils import property

from openvino._pyopenvino.properties import supported_properties as sp
from openvino._pyopenvino.properties import cache_dir as cd


class Property(str):
    def __new__(cls, prop):
        instance = super().__new__(cls, prop())
        instance.prop = prop
        return instance

    def __call__(self, *args):
        if args is not None:
            return self.prop(*args)
        return self.prop()


@property
def _supported_properties():
    return Property(sp)


@property
def _cache_dir():
    return Property(cd)
