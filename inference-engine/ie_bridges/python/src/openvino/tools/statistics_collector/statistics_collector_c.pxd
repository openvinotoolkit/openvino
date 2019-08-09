from libc.stddef cimport size_t
from libcpp.string cimport string


cdef extern from "<statistics_processor.hpp>":

    cdef struct ct_preprocessingOptions:
        string _pp_type
        size_t _pp_size
        size_t _pp_width
        size_t _pp_height

    cdef cppclass StatisticsCollector:
        StatisticsCollector(const string& deviceName,
                            const string& custom_cpu_library,
                            const string& custom_cldnn,
                            const string& modelFilePath,
                            const string& imagesPath,
                            size_t img_number,
                            size_t batch,
                            const ct_preprocessingOptions& preprocessingOptions,
                            const string& progress) except +
        void collectStatisticsToIR(const string& outModelName, const string& output_precision)
        ct_preprocessingOptions ppOptions
