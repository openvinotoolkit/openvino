from .cimport ie_api_impl_defs as C
from .ie_api_impl_defs cimport Blob, TensorDesc

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr

cdef class BlobBuffer:
    cdef Blob.Ptr ptr
    cdef char*format
    cdef vector[Py_ssize_t] shape
    cdef vector[Py_ssize_t] strides
    cdef reset(self, Blob.Ptr &)
    cdef char*_get_blob_format(self, const TensorDesc & desc)

    cdef public:
        total_stride, item_size

cdef class InferRequest:
    cdef C.InferRequestWrap *impl

    cpdef BlobBuffer _get_blob_buffer(self, const string & blob_name)

    cpdef infer(self, inputs = ?)
    cpdef async_infer(self, inputs = ?)
    cpdef wait(self, timeout = ?)
    cpdef get_perf_counts(self)
    cdef void user_callback(self, int status) with gil
    cdef public:
        _inputs_list, _outputs_list, _py_callback, _py_data, _py_callback_used, _py_callback_called

cdef class IENetwork:
    cdef C.IENetwork impl

cdef class ExecutableNetwork:
    cdef unique_ptr[C.IEExecNetwork] impl
    cdef C.IEPlugin plugin_impl
    cdef C.IECore ie_core_impl
    cdef public:
        _requests, _infer_requests, inputs, outputs

cdef class IEPlugin:
    cdef C.IEPlugin impl
    cpdef ExecutableNetwork load(self, IENetwork network, int num_requests = ?, config = ?)
    cpdef void set_config(self, config)
    cpdef void add_cpu_extension(self, str extension_path) except *
    cpdef void set_initial_affinity(self, IENetwork network) except *
    cpdef set get_supported_layers(self, IENetwork net)

cdef class IENetLayer:
    cdef C.IENetLayer impl

cdef class InputInfo:
    cdef C.InputInfo impl

cdef class OutputInfo:
    cdef C.OutputInfo impl

cdef class LayersStatsMap(dict):
    cdef C.IENetwork net_impl

cdef class IECore:
    cdef C.IECore impl
    cpdef ExecutableNetwork load_network(self, IENetwork network, str device_name, config = ?, int num_requests = ?)

cdef class DataPtr:
    cdef shared_ptr[C.Data] _ptr
