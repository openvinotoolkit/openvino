# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.utils import property

from openvino._pyopenvino.properties import supported_properties as sp
from openvino._pyopenvino.properties import cache_dir as cd

class A(str):
    @classmethod
    def __call__(cls, *args):
        if args is not None:
            return sp(*args)
        return sp()
    @classmethod
    def __str__(cls):
        return sp()
    @classmethod
    def __repr__(cls):
        return sp()


@property
def _supported_properties():
    return A()

class B(str):
    @classmethod
    def __call__(cls, *args):
        if args is not None:
            return cd(*args)
        return cd()
    @classmethod
    def __str__(cls):
        return str("\'" + cd() + "\'")
    @classmethod
    def __repr__(cls):
        return str("\'" + cd() + "\'")


@property
def _cache_dir():
    return B()
