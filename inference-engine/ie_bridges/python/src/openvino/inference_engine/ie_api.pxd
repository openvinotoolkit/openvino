# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cimport ie_api_impl_defs as C
from .ie_api_impl_defs cimport CBlob, CTensorDesc, InputInfo, CPreProcessChannel, CPreProcessInfo, CExecutableNetwork, CVariableState

import os

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr

cdef class Blob:
    cdef CBlob.Ptr _ptr
    cdef object _is_const
    cdef public object _array_data
    cdef public object _initial_shape

cdef class BlobBuffer:
    cdef CBlob.Ptr ptr
    cdef char*format
    cdef vector[Py_ssize_t] shape
    cdef vector[Py_ssize_t] strides
    cdef reset(self, CBlob.Ptr &, vector[size_t] representation_shape = ?)
    cdef char*_get_blob_format(self, const CTensorDesc & desc)

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
        _inputs_list, _outputs_list, _py_callback, _py_data, _py_callback_used, _py_callback_called, _user_blobs

cdef class IENetwork:
    cdef C.IENetwork impl
    cdef shared_ptr[CExecutableNetwork] _ptr_plugin

cdef class ExecutableNetwork:
    cdef unique_ptr[C.IEExecNetwork] impl
    cdef C.IECore ie_core_impl
    cpdef wait(self, num_requests = ?, timeout = ?)
    cpdef get_idle_request_id(self)
    cdef public:
        _requests, _infer_requests

cdef class IECore:
    cdef C.IECore impl
    cpdef IENetwork read_network(self, model : [str, bytes, os.PathLike],
                                 weights : [str, bytes, os.PathLike] = ?, bool init_from_buffer = ?)
    cpdef ExecutableNetwork load_network(self, network: [IENetwork, str],
                                         str device_name, config = ?, int num_requests = ?)
    cpdef ExecutableNetwork import_network(self, str model_file, str device_name, config = ?, int num_requests = ?)


cdef class DataPtr:
    cdef C.DataPtr _ptr
    cdef C.IENetwork * _ptr_network
    cdef shared_ptr[CExecutableNetwork] _ptr_plugin

cdef class CDataPtr:
    cdef C.CDataPtr _ptr
    cdef shared_ptr[CExecutableNetwork] _ptr_plugin

cdef class TensorDesc:
    cdef C.CTensorDesc impl

cdef class InputInfoPtr:
    cdef InputInfo.Ptr _ptr
    cdef C.IENetwork * _ptr_network

cdef class InputInfoCPtr:
    cdef InputInfo.CPtr _ptr
    cdef shared_ptr[CExecutableNetwork] _ptr_plugin

cdef class PreProcessInfo:
    cdef CPreProcessInfo* _ptr
    cdef const CPreProcessInfo* _cptr
    cpdef object _user_data

cdef class PreProcessChannel:
    cdef CPreProcessChannel.Ptr _ptr

cdef class VariableState:
    cdef C.CVariableState impl
