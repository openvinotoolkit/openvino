# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cimport offline_transformations_api_impl_defs as C
from ..inference_engine.ie_api cimport IENetwork
from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.string cimport string
from libc.stdint cimport int64_t
from libcpp.map cimport map

import numpy as np


def ApplyMOCTransformations(IENetwork network, bool cf):
    C.ApplyMOCTransformations(network.impl, cf)


def ApplyPOTTransformations(IENetwork network, string device):
    C.ApplyPOTTransformations(network.impl, device)


def ApplyLowLatencyTransformation(IENetwork network, bool use_const_initializer = True):
    C.ApplyLowLatencyTransformation(network.impl, use_const_initializer)


def ApplyPruningTransformation(IENetwork network):
    C.ApplyPruningTransformation(network.impl)


def GenerateMappingFile(IENetwork network, string path, bool extract_names):
    C.GenerateMappingFile(network.impl, path, extract_names)


cdef class ConstantInfo:
    cdef C.ConstantInfoPtr info

    def __cinit__(self, data: np.ndarray, axis: int, shape_size: int):
        self.info = C.CreateConstantInfo(data, axis, shape_size)

    property data:
        def __get__(self):
            return deref(self.info).data

        def __set__(self, d: np.ndarray):
            deref(self.info).data = d

    property axis:
        def __get__(self):
            return deref(self.info).axis

        def __set__(self, int a):
            deref(self.info).axis = a

    property shape_size:
        def __get__(self):
            return deref(self.info).shape_size

        def __set__(self, int s):
            deref(self.info).shape_size = s


def ApplyScaleInputs(IENetwork network, values: dict[string, ConstantInfo]):
    cdef map[string, C.ConstantInfoPtr] mapped
    for key, value in values.items():
        mapped[key.encode()] = C.CreateConstantInfo(value.data, value.axis, value.shape_size)
    C.ApplyScaleInputs(network.impl, mapped)


def ApplySubtractMeanInputs(IENetwork network, values: dict[string, ConstantInfo]):
    cdef map[string, C.ConstantInfoPtr] mapped
    for key, value in values.items():
        mapped[key.encode()] = C.CreateConstantInfo(value.data, value.axis, value.shape_size)
    C.ApplySubtractMeanInputs(network.impl, mapped)


def CheckAPI():
    C.CheckAPI()
