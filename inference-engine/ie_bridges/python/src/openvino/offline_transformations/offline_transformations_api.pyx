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


def ApplyLowLatencyTransformation(IENetwork network, bool use_const_initializer = True, sub_graph_iterations = None):
    cdef map[string, int64_t] c_sub_graph_iterations

    for k, v in sub_graph_iterations.items():
        c_sub_graph_iterations[k.encode()] = v
    C.ApplyLowLatencyTransformation(network.impl, use_const_initializer, c_sub_graph_iterations)


def ApplyPruningTransformation(IENetwork network):
    C.ApplyPruningTransformation(network.impl)


def GenerateMappingFile(IENetwork network, string path, bool extract_names):
    C.GenerateMappingFile(network.impl, path, extract_names)


def CheckAPI():
    C.CheckAPI()
