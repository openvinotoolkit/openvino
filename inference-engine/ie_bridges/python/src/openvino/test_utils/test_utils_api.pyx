# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cimport test_utils_api_impl_defs as C
from ..inference_engine.ie_api cimport IENetwork

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.pair cimport pair


def CompareNetworks(IENetwork lhs, IENetwork rhs):
    cdef pair[bool, string] c_pair
    c_pair = C.CompareNetworks(lhs.impl, rhs.impl)
    return c_pair
