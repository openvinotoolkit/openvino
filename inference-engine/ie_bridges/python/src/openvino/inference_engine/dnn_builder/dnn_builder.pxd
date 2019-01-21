from .cimport dnn_builder_impl_defs as C
from libcpp.memory cimport shared_ptr

cdef class NetworkBuilder:
    cdef C.NetworkBuilder impl

cdef class INetwork:
    cdef C.INetwork impl

cdef class ILayer:
    cdef C.ILayer impl

cdef class Port:
    cdef C.Port impl

cdef class PortInfo:
    cdef C.PortInfo impl

cdef class Connection:
    cdef C.Connection impl

cdef class LayerBuilder:
    cdef C.LayerBuilder impl

cdef class LayerConstantData(dict):
    cdef shared_ptr[C.LayerBuilder] impl