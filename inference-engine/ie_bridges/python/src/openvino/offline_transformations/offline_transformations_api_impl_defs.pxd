# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from ..inference_engine.ie_api_impl_defs cimport IENetwork

cdef extern from "offline_transformations_api_impl.hpp" namespace "InferenceEnginePython":
    cdef void ApplyMOCTransformations(IENetwork network, bool cf)

    cdef void ApplyPOTTransformations(IENetwork network, string device)

    cdef void ApplyLowLatencyTransformation(IENetwork network, bool use_const_initializer)

    cdef void ApplyPruningTransformation(IENetwork network)

    cdef void GenerateMappingFile(IENetwork network, string path, bool extract_names)

    cdef cppclass ConstantInfo:
        ConstantInfo(vector[float], int, int) except +
        vector[float] data;
        int axis;
        int shape_size;

    ctypedef shared_ptr[ConstantInfo] ConstantInfoPtr

    cdef ConstantInfoPtr CreateConstantInfo(vector[float], int, int)

    cdef ConstantInfoPtr CreateEmptyConstantInfo()

    cdef void ApplyScaleInputs(IENetwork network, const map[string, ConstantInfoPtr]& values)

    cdef void ApplySubtractMeanInputs(IENetwork network, const map[string, ConstantInfoPtr]& values)

    cdef void CheckAPI()
