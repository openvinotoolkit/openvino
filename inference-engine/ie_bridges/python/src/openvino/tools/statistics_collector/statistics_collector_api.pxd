from .cimport statistics_collector_c as C
from libcpp.string cimport string


cdef class StatisticsCollector:
    cdef C.StatisticsCollector* _impl
    cdef C.ct_preprocessingOptions ppOptions
    cpdef void collectStatisticsToIR(self, str outModelName, str output_precision)