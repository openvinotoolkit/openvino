from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr
from libcpp.map cimport map
from ..ie_api_impl_defs cimport IENetwork

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
        const char*name() const

cdef extern from "dnn_builder_impl.hpp" namespace "InferenceEnginePython":
    cdef cppclass ILayer:
        const string name
        size_t id
        string type
        map[string, string] parameters
        vector[Port] in_ports
        vector[Port] out_ports
        map[string, Blob.Ptr] constant_data;


    cdef cppclass INetwork:
        string name
        size_t size
        vector[ILayer] layers
        vector[ILayer] inputs
        vector[ILayer] outputs
        vector[Port] in_ports;
        vector[Port] out_ports;
        vector[Connection] getLayerConnections(size_t layer_id);
        IENetwork to_ie_network();

    cdef cppclass NetworkBuilder:
        NetworkBuilder() except +
        NetworkBuilder(string name) except +
        NetworkBuilder from_ie_network(IENetwork &icnn_net) except +
        INetwork build() except +
        vector[LayerBuilder] getLayers() except +
        LayerBuilder getLayer(size_t layer_id) except +
        void removeLayer(const LayerBuilder& layer) except +
        const vector[Connection] getLayerConnections(const LayerBuilder& layer) except +
        void disconnect(const Connection& connection) except +
        void connect(const PortInfo& input, const PortInfo& output) except +
        size_t addLayer(const LayerBuilder& layer) except +
        size_t addAndConnectLayer(const vector[PortInfo]& input, const LayerBuilder& layer);

    cdef cppclass Port:
        Port() except +
        Port(const vector[size_t] & shapes) except +
        const vector[size_t] shape


    cdef cppclass PortInfo:
        PortInfo(size_t layer_id, size_t port_id) except +
        PortInfo() except +
        size_t layer_id
        size_t port_id

    cdef cppclass Connection:
        Connection(PortInfo input, PortInfo output) except +
        Connection() except +
        PortInfo _from
        PortInfo to

    cdef cppclass LayerBuilder:
        LayerBuilder()
        LayerBuilder(const string& type, const string& name ) except +
        size_t id
        LayerBuilder from_ilayer(const ILayer& ilayer) except +
        string getName() except +
        string getType() except +
        vector[Port] getInputPorts() except +
        vector[Port] getOutputPorts() except +
        map[string, string] getParameters() except +
        void setParameters(map[string, string] params_map) except +
        void setName(const string & name) except +
        void setType(const string & type) except +
        void setInputPorts(const vector[Port] ports) except +
        void setOutputPorts(const vector[Port] ports) except +
        ILayer build() except +
        map[string, Blob.Ptr] getConstantData()
        void setConstantData(map[string, Blob.Ptr] &const_data)
        # TODO: Fix LAyerBuilder object copying - pass by reference
        # void addConstantData(const string & name, Blob.Ptr data)
        Blob.Ptr allocateBlob(vector[size_t] dims, const string & precision)
