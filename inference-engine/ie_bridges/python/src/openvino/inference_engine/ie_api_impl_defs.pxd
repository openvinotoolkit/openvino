from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr, shared_ptr, weak_ptr
from libc.stdint cimport int64_t, uint8_t


cdef extern from "<inference_engine.hpp>" namespace "InferenceEngine":
    ctypedef vector[size_t] SizeVector


    cdef cppclass TensorDesc:
        SizeVector& getDims()
        const Precision& getPrecision() const

    cdef cppclass Data:
        const Precision getPrecision() const
        void setPrecision(const Precision& precision) const
        const SizeVector getDims()
        const string& getName() const
        const Layout getLayout() const
        void setLayout(Layout layout) const
        const bool isInitialized() const
        weak_ptr[CNNLayer] & getCreatorLayer()
        map[string, shared_ptr[CNNLayer]] & getInputTo()

    ctypedef shared_ptr[Data] DataPtr
    ctypedef weak_ptr[Data] DataWeakPtr
    ctypedef shared_ptr[const Data] CDataPtr

    cdef cppclass Blob:
        ctypedef shared_ptr[Blob] Ptr
        const TensorDesc& getTensorDesc() const
        size_t element_size()  const

    cdef cppclass Precision:
        const char*name() const
        @staticmethod
        const Precision FromStr(const string& str)

    cdef cppclass CNNLayer:
        string name
        string type
        Precision precision
        vector[DataPtr] outData
        vector[DataWeakPtr] insData
        string affinity
        map[string, string] params
        map[string, Blob.Ptr] blobs

    ctypedef weak_ptr[CNNLayer] CNNLayerWeakPtr
    ctypedef shared_ptr[CNNLayer] CNNLayerPtr


    cdef struct apiVersion:
        int minor
        int major

    cdef cppclass Version:
        const char *buildNumber
        const char *description
        apiVersion apiVersion

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

#    cdef cppclass IENetLayer:
#        string layout
#        vector[string] children
#        vector[string] parents
#        void setAffinity(const string & target_affinity) except +
#        void setParams(const map[string, string] & params_map) except +
#        map[string, Blob.Ptr] getWeights() except +
#        void setPrecision(string precision) except +
#        vector[DataPtr] getOutData() except +


    cdef cppclass ProfileInfo:
        string status
        string exec_type
        string layer_type
        long long real_time
        long long cpu_time
        unsigned int execution_index

    cdef cppclass WeightsInfo:
        Blob.Ptr & weights;
        Blob.Ptr & biases;
        map[string, Blob.Ptr] custom_blobs;

    cdef cppclass IEExecNetwork:
        vector[InferRequestWrap] infer_requests
        IENetwork GetExecGraphInfo() except +
        map[string, DataPtr] getInputs()
        map[string, CDataPtr] getOutputs()
        void exportNetwork(const string & model_file) except +
        object getMetric(const string & metric_name)
        object getConfig(const string & metric_name)
        int wait(int num_requests, int64_t timeout)
        int getIdleRequestId()

    cdef cppclass IENetwork:
        IENetwork() except +
        IENetwork(object) except +
        IENetwork(const string &, const string &) except +
        string name
        size_t batch_size
        string precision
        map[string, vector[size_t]] inputs
        const vector[CNNLayerPtr] getLayers() except +
        map[string, DataPtr] getInputs() except +
        map[string, DataPtr] getOutputs() except +
        void addOutput(string &, size_t) except +
        void setAffinity(map[string, string] & types_affinity_map, map[string, string] & layers_affinity_map) except +
        void setBatch(size_t size) except +
        size_t getBatch() except +
        void setLayerParams(map[string, map[string, string]] params_map) except +
        void serialize(const string& path_to_xml, const string& path_to_bin) except +
        void reshape(map[string, vector[size_t]] input_shapes) except +
        void setStats(map[string, map[string, vector[float]]] & stats) except +
        map[string, map[string, vector[float]]] getStats() except +
        void load_from_buffer(const char*xml, size_t xml_size, uint8_t*bin, size_t bin_size) except +
        object getFunction() except +

    cdef cppclass IEPlugin:
        IEPlugin() except +
        IEPlugin(const string &, const vector[string] &) except +
        unique_ptr[IEExecNetwork] load(IENetwork & net, int num_requests, const map[string, string]& config) except +
        void addCpuExtension(const string &) except +
        void setConfig(const map[string, string] &) except +
        void setInitialAffinity(IENetwork & net) except +
        set[string] queryNetwork(const IENetwork & net) except +
        string device_name
        string version

    cdef cppclass InferRequestWrap:
        double exec_time;
        int index;
        void getBlobPtr(const string & blob_name, Blob.Ptr & blob_ptr) except +
        map[string, ProfileInfo] getPerformanceCounts() except +
        void infer() except +
        void infer_async() except +
        int wait(int64_t timeout) except +
        void setBatch(int size) except +
        void setCyCallback(void (*)(void*, int), void *) except +

    cdef cppclass IECore:
        IECore() except +
        IECore(const string & xml_config_file) except +
        map[string, Version] getVersions(const string & deviceName) except +
        IENetwork readNetwork(const string& modelPath, const string& binPath) except +
        IENetwork readNetwork(const string& modelPath,uint8_t*bin, size_t bin_size) except +
        unique_ptr[IEExecNetwork] loadNetwork(IENetwork network, const string deviceName,
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

    cdef T*get_buffer[T](Blob &)

    cdef string get_version()
