# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr, shared_ptr, weak_ptr
from libc.stdint cimport int64_t, uint8_t


cdef extern from "<inference_engine.hpp>" namespace "InferenceEngine":
    ctypedef vector[size_t] SizeVector

    cdef cppclass CExecutableNetwork "InferenceEngine::ExecutableNetwork"
    
    cdef cppclass TBlob[T]:
        ctypedef shared_ptr[TBlob[T]] Ptr

    cdef cppclass CBlob "InferenceEngine::Blob":
        ctypedef shared_ptr[CBlob] Ptr
        const CTensorDesc& getTensorDesc()  except +
        size_t element_size()  except +
        void allocate()

    cdef TBlob[Type].Ptr make_shared_blob[Type](const CTensorDesc& tensorDesc)

    cdef TBlob[Type].Ptr make_shared_blob[Type](const CTensorDesc& tensorDesc, Type* ptr, size_t size)

    cdef cppclass CTensorDesc "InferenceEngine::TensorDesc":
        CTensorDesc() except +
        CTensorDesc(const Precision& precision, SizeVector dims, Layout layout) except +
        SizeVector& getDims() except +
        void setDims(const SizeVector& dims) except +
        Layout getLayout() except +
        void setLayout(Layout l) except +
        const Precision& getPrecision() except +
        void setPrecision(const Precision& p) except +


    cdef cppclass Data:
        const Precision getPrecision() const
        void setPrecision(const Precision& precision) const
        const SizeVector getDims() except +
        const string& getName() except +
        const Layout getLayout() except +
        void setLayout(Layout layout) except +
        const bool isInitialized() except +

    ctypedef shared_ptr[Data] DataPtr
    ctypedef weak_ptr[Data] DataWeakPtr
    ctypedef shared_ptr[const Data] CDataPtr

    cdef cppclass InputInfo:
        ctypedef shared_ptr[InputInfo] Ptr
        ctypedef shared_ptr[const InputInfo] CPtr
        Precision getPrecision() const
        void setPrecision(Precision p)
        Layout getLayout()
        void setLayout(Layout l)
        const string& name() const
        DataPtr getInputData() const
        CPreProcessInfo& getPreProcess()
        const CTensorDesc& getTensorDesc() const
        void setInputData(DataPtr inputPtr)


    cdef cppclass CPreProcessChannel "InferenceEngine::PreProcessChannel":
        ctypedef shared_ptr[CPreProcessChannel] Ptr
        CBlob.Ptr meanData
        float stdScale
        float meanValue

    cdef cppclass CPreProcessInfo "InferenceEngine::PreProcessInfo":
        CPreProcessChannel.Ptr& operator[](size_t index)
        size_t getNumberOfChannels() const
        void init(const size_t numberOfChannels)
        void setMeanImage(const CBlob.Ptr& meanImage)
        void setMeanImageForChannel(const CBlob.Ptr& meanImage, const size_t channel)
        vector[CPreProcessChannel.Ptr] _channelsInfo
        ColorFormat getColorFormat() const
        void setColorFormat(ColorFormat fmt)
        ResizeAlgorithm getResizeAlgorithm() const
        void setResizeAlgorithm(const ResizeAlgorithm& alg)
        MeanVariant getMeanVariant() const
        void setVariant(const MeanVariant& variant)

    ctypedef map[string, InputInfo.CPtr] InputsDataMap

    cdef cppclass Precision:
        const char*name() const
        @staticmethod
        const Precision FromStr(const string& str)

    cdef struct apiVersion:
        int minor
        int major

    cdef cppclass Version:
        const char *buildNumber
        const char *description
        apiVersion apiVersion

    cpdef enum MeanVariant:
        pass

    cpdef enum ResizeAlgorithm:
        pass

    cpdef enum ColorFormat:
        pass

    cdef enum Layout:
        ANY
        NCHW
        NHWC
        NCDHW
        NDHWC
        OIHW
        GOIHW
        OIDHW
        GOIDHW
        SCALAR
        C
        CHW
        HW
        NC
        CN
        BLOCKED


cdef extern from "ie_api_impl.hpp" namespace "InferenceEnginePython":

    cdef cppclass CVariableState:
        void reset() except +
        string getName() except +
        CBlob.Ptr getState() except +
        void setState(CBlob.Ptr state) except +

    cdef cppclass ProfileInfo:
        string status
        string exec_type
        string layer_type
        long long real_time
        long long cpu_time
        unsigned int execution_index

    cdef cppclass WeightsInfo:
        CBlob.Ptr & weights;
        CBlob.Ptr & biases;
        map[string, CBlob.Ptr] custom_blobs;

    cdef cppclass IEExecNetwork:
        vector[InferRequestWrap] infer_requests
        IENetwork GetExecGraphInfo() except +
        map[string, DataPtr] getInputs() except +
        map[string, CDataPtr] getOutputs() except +
        map[string, InputInfo.CPtr] getInputsInfo()
        void exportNetwork(const string & model_file) except +
        object getMetric(const string & metric_name) except +
        object getConfig(const string & metric_name) except +
        int wait(int num_requests, int64_t timeout)
        int getIdleRequestId()
        shared_ptr[CExecutableNetwork] getPluginLink() except +

    cdef cppclass IENetwork:
        IENetwork() except +
        IENetwork(object) except +
        IENetwork(const string &, const string &) except +
        string name
        size_t batch_size
        string precision
        map[string, vector[size_t]] inputs
        const map[string, InputInfo.Ptr] getInputsInfo() except +
        const map[string, DataPtr] getInputs() except +
        map[string, DataPtr] getOutputs() except +
        void addOutput(string &, size_t) except +
        void setAffinity(map[string, string] & types_affinity_map, map[string, string] & layers_affinity_map) except +
        void setBatch(size_t size) except +
        size_t getBatch() except +
        void setLayerParams(map[string, map[string, string]] params_map) except +
        void serialize(const string& path_to_xml, const string& path_to_bin) except +
        void reshape(map[string, vector[size_t]] input_shapes) except +
        void load_from_buffer(const char*xml, size_t xml_size, uint8_t*bin, size_t bin_size) except +
        object getFunction() except +
        void convertToOldRepresentation() except +
        string getOVNameForTensor(const string &) except +

    cdef cppclass InferRequestWrap:
        double exec_time;
        int index;
        CBlob.Ptr getBlobPtr(const string & blob_name) except +
        void setBlob(const string & blob_name, const CBlob.Ptr & blob_ptr) except +
        void setBlob(const string &blob_name, const CBlob.Ptr &blob_ptr, CPreProcessInfo& info) except +
        const CPreProcessInfo& getPreProcess(const string& blob_name) except +
        map[string, ProfileInfo] getPerformanceCounts() except +
        void infer() except +
        void infer_async() except +
        int wait(int64_t timeout) except +
        void setBatch(int size) except +
        void setCyCallback(void (*)(void*, int), void *) except +
        vector[CVariableState] queryState() except +

    cdef cppclass IECore:
        IECore() except +
        IECore(const string & xml_config_file) except +
        map[string, Version] getVersions(const string & deviceName) except +
        IENetwork readNetwork(const string& modelPath, const string& binPath) except +
        IENetwork readNetwork(const string& modelPath,uint8_t*bin, size_t bin_size) except +
        unique_ptr[IEExecNetwork] loadNetwork(IENetwork network, const string deviceName,
                                              const map[string, string] & config, int num_requests) except +
        unique_ptr[IEExecNetwork] loadNetworkFromFile(const string & modelPath, const string & deviceName,
                                              const map[string, string] & config, int num_requests) except +
        unique_ptr[IEExecNetwork] importNetwork(const string & modelFIle, const string & deviceName,
                                                const map[string, string] & config, int num_requests) except +
        map[string, string] queryNetwork(IENetwork network, const string deviceName,
                                         const map[string, string] & config) except +
        void setConfig(const map[string, string] & config, const string & deviceName) except +
        void registerPlugin(const string & pluginName, const string & deviceName) except +
        void unregisterPlugin(const string & deviceName) except +
        void registerPlugins(const string & xmlConfigFile) except +
        void addExtension(const string & ext_lib_path, const string & deviceName) except +
        vector[string] getAvailableDevices() except +
        object getMetric(const string & deviceName, const string & name) except +
        object getConfig(const string & deviceName, const string & name) except +

    cdef T*get_buffer[T](CBlob &)

    cdef string get_version()

    cdef IENetwork read_network(string path_to_xml, string path_to_bin)
