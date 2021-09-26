# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cimport offline_transformations_api_impl_defs as C
from ..inference_engine.ie_api cimport IENetwork

from libcpp cimport bool
from libcpp.string cimport string
from libc.stdint cimport int64_t


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


def CheckAPI():
    C.CheckAPI()
