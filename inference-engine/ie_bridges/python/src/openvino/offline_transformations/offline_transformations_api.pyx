# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cimport offline_transformations_api_impl_defs as C
from ..inference_engine.ie_api cimport IENetwork

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport int64_t


def ApplyMOCTransformations(IENetwork network, bool cf):
    C.ApplyMOCTransformations(network.impl, cf)


def ApplyPOTTransformations(IENetwork network, string device):
    C.ApplyPOTTransformations(network.impl, device)


def ApplyMakeStatefulTransformation(IENetwork network, param_res_names : dict):
    cdef map[string, string] c_param_res_names
    for param_name, res_name in param_res_names.items():
        if type(param_name) != str or type(res_name) != str:
            raise TypeError("Only string keys and values are allowed!")
        c_param_res_names[param_name.encode()] = res_name.encode()
    C.ApplyMakeStatefulTransformation(network.impl, c_param_res_names)


def ApplyLowLatencyTransformation(IENetwork network, bool use_const_initializer = True):
    C.ApplyLowLatencyTransformation(network.impl, use_const_initializer)


def ApplyPruningTransformation(IENetwork network):
    C.ApplyPruningTransformation(network.impl)


def GenerateMappingFile(IENetwork network, string path, bool extract_names):
    C.GenerateMappingFile(network.impl, path, extract_names)


def CheckAPI():
    C.CheckAPI()
