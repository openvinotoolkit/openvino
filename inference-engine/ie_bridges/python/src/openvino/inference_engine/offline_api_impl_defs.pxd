from libcpp cimport bool
from .ie_api_impl_defs cimport IENetwork

cdef extern from "offline_api_impl.hpp" namespace "InferenceEnginePython":
    cdef void ApplyMOCTransformations(IENetwork network, bool cf)