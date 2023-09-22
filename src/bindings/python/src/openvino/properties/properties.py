# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.utils import module_property

from openvino._pyopenvino.properties import supported_properties as sp


class A:
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


@module_property
def _supported_properties():
    return A()
