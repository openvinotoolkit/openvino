# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map

from ..inference_engine.ie_api_impl_defs cimport IENetwork

cdef extern from "offline_transformations_api_impl.hpp" namespace "InferenceEnginePython":
    cdef void ApplyMOCTransformations(IENetwork network, bool cf)

    cdef void ApplyPOTTransformations(IENetwork network, string device)

    cdef void ApplyLowLatencyTransformation(IENetwork network, bool use_const_initializer)

    cdef void ApplyMakeStatefulTransformation(IENetwork network, map[string, string]& in_out_names)

    cdef void ApplyPruningTransformation(IENetwork network)

    cdef void GenerateMappingFile(IENetwork network, string path, bool extract_names)

    cdef void Serialize(IENetwork network, string path_to_xml, string path_to_bin)

    cdef void CheckAPI()
