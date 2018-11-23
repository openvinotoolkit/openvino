# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from libc.stddef cimport size_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp cimport bool
from libc.stdint cimport int64_t



cdef extern from "<inference_engine.hpp>" namespace "InferenceEngine":
    ctypedef vector[size_t] SizeVector

    cdef cppclass TensorDesc:
        SizeVector& getDims()
        const Precision& getPrecision() const

    cdef cppclass Blob:
        ctypedef shared_ptr[Blob] Ptr
        const TensorDesc& getTensorDesc() const
        size_t element_size()  const

    cdef cppclass Precision:
        const char* name() const


cdef extern from "ie_api_impl.hpp" namespace "InferenceEnginePython":
    cdef cppclass IENetLayer:
        string name
        string type
        string precision
        string affinity
        map[string, string] params
        # map[string, BlobInfo] blob_info
        # map[string, Blob.Ptr] weights;
        void setAffinity(const string & target_affinity) except +
        void setParams(const map[string, string] & params_map) except +
        map[string, Blob.Ptr] getWeights() except +
        void setPrecision(string precision) except +

    cdef cppclass InputInfo:
        vector[size_t] dims
        string precision
        string layout
        void setPrecision(string precision)
        void setLayout(string layout)

    cdef cppclass OutputInfo:
        vector[size_t] dims
        string precision
        string layout
        void setPrecision(string precision)


    cdef cppclass ProfileInfo:
        string status
        string exec_type
        string layer_type
        long long real_time
        long long cpu_time
        unsigned int execution_index

    cdef cppclass WeightsInfo:
        Blob.Ptr &weights;
        Blob.Ptr &biases;
        map[string, Blob.Ptr] custom_blobs;


    cdef cppclass IEExecNetwork:
        vector[InferRequestWrap] infer_requests

    cdef cppclass IENetwork:
        string name
        size_t batch_size
        map[string, vector[size_t]] inputs
        map[string, IENetLayer] getLayers() except +
        map[string, InputInfo] getInputs() except +
        map[string, OutputInfo] getOutputs() except +
        void addOutputs(vector[string] &, string &) except +
        void setAffinity(map[string, string] &types_affinity_map, map[string, string] &layers_affinity_map) except +
        void setBatch(size_t size) except +
        void setLayerParams(map[string, map[string, string]] params_map) except +
        void reshape(map[string, vector[size_t]] input_shapes) except +

    cdef cppclass IEPlugin:
        IEPlugin() except +
        IEPlugin(const string &, const vector[string] &) except +
        unique_ptr[IEExecNetwork] load(IENetwork & net, int num_requests, const map[string, string]& config) except +
        void addCpuExtension(const string &) except +
        void setConfig(const map[string, string]&) except +
        void setInitialAffinity(IENetwork & net) except +
        set[string] queryNetwork(const IENetwork &net) except +
        string device_name
        string version

    cdef cppclass IENetReader:
        IENetwork read(const string &, const string &) except +

    cdef cppclass InferRequestWrap:
        vector[string] getInputsList() except +
        vector[string] getOutputsList() except +
        Blob.Ptr& getOutputBlob(const string &blob_name) except +
        Blob.Ptr& getInputBlob(const string &blob_name) except +
        map[string, ProfileInfo] getPerformanceCounts() except +
        void infer() except +
        void infer_async() except +
        int wait(int64_t timeout) except +

    cdef T* get_buffer[T](Blob &)

    cdef string get_version()
