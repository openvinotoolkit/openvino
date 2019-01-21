# #distutils: language=c++
#from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from ..ie_api cimport IENetwork, BlobBuffer
from .cimport dnn_builder_impl_defs as C
from .dnn_builder_impl_defs cimport Blob
import numpy as np


np_precision_map = {
            "float32": "FP32",
            "float16": "FP16",
            "int32": "I32",
            "int16": "I16",
            "uint16": "U16",
            "int8": "I8",
            "uint8": "U8",
        }
cdef class NetworkBuilder:
    def __cinit__(self, name=None, IENetwork ie_net=None):
        if name is not None and ie_net is not None:
            raise AttributeError("Both name and ie_net arguments are defined")
        elif name is not None:
            self.impl = C.NetworkBuilder(name.encode())
        elif ie_net is not None:
            self.impl = C.NetworkBuilder().from_ie_network(ie_net.impl)

    def build(self):
        cdef INetwork i_net = INetwork()
        i_net.impl = self.impl.build()
        return i_net

    def get_layer(self, id: int):
        cdef LayerBuilder py_layer = LayerBuilder()
        py_layer.impl = self.impl.getLayer(id)
        return py_layer

    @property
    def layers(self):
        cdef vector[C.LayerBuilder] c_layers = self.impl.getLayers()
        cdef LayerBuilder py_layer
        py_layers = {}
        for l in c_layers:
            py_layer = LayerBuilder()
            py_layer.impl = l
            py_layers[l.getName().decode()] = py_layer
        return py_layers

    def remove_layer(self, LayerBuilder layer):
        self.impl.removeLayer(layer.impl)

    def get_layer_connection(self, LayerBuilder layer):
        cdef vector[C.Connection] c_connections = self.impl.getLayerConnections(layer.impl)
        cdef Connection connection
        connections = []
        for con in c_connections:
            connection = Connection()
            connection.impl = con
            connections.append(connection)
        return connections

    def disconnect(self, Connection connection):
        self.impl.disconnect(connection.impl)

    def connect(self, PortInfo input, PortInfo output):
        self.impl.connect(input.impl, output.impl)

    def add_layer(self, LayerBuilder layer, input_ports: list = None):
        cdef vector[C.PortInfo] c_ports
        cdef PortInfo c_port
        if not input_ports:
            return self.impl.addLayer(layer.impl)
        else:
            for p in input_ports:
                c_port = PortInfo(p.layer_id, p.port_id)
                c_ports.push_back(c_port.impl)
            return self.impl.addAndConnectLayer(c_ports, layer.impl)

cdef class INetwork:
    def __iter__(self):
        cdef ILayer layer
        layers = []
        cdef vector[C.ILayer] c_layers = self.impl.layers
        for l in c_layers:
            layer = ILayer()
            layer.impl = l
            layers.append(layer)
        return iter(layers)

    @property
    def layers(self):
        cdef ILayer layer
        layers = {}
        cdef vector[C.ILayer] c_layers = self.impl.layers
        for l in c_layers:
            layer = ILayer()
            layer.impl = l
            layers[l.name.decode()] = layer
        return layers

    @property
    def inputs(self):
        cdef ILayer layer
        layers = {}
        cdef vector[C.ILayer] c_layers = self.impl.inputs
        for l in c_layers:
            layer = ILayer()
            layer.impl = l
            layers[l.name.decode()] = layer
        return layers

    @property
    def outputs(self):
        cdef ILayer layer
        layers = {}
        cdef vector[C.ILayer] c_layers = self.impl.outputs
        for l in c_layers:
            layer = ILayer()
            layer.impl = l
            layers[l.name.decode()] = layer
        return layers

    @property
    def name(self):
        return self.impl.name.decode()


    @property
    def size(self):
        return self.impl.size

    def get_layer_connection(self, layer: ILayer):
        cdef Connection connection
        connections = []
        cdef vector[C.Connection] c_connections = self.impl.getLayerConnections(layer.id)
        for con in c_connections:
            connection = Connection()
            connection.impl = con
            connections.append(connection)
        return connections

    def to_ie_network(self):
        cdef IENetwork net = IENetwork()
        net.impl = self.impl.to_ie_network()
        return net

cdef class ILayer:
    @property
    def name(self):
        return self.impl.name.decode()

    @property
    def id(self):
        return self.impl.id

    @property
    def type(self):
        return self.impl.type.decode()

    @property
    def params(self):
        return {k.decode(): v.decode() for k, v in self.impl.parameters}

    @property
    def input_ports(self):
        cdef Port port
        cdef vector[C.Port] c_ports = self.impl.in_ports
        ports = []
        for p in c_ports:
            port = Port()
            port.impl = p
            ports.append(port)
        return ports

    @property
    def output_ports(self):
        cdef Port port
        cdef vector[C.Port] c_ports = self.impl.out_ports
        ports = []
        for p in c_ports:
            port = Port()
            port.impl = p
            ports.append(port)
        return ports

    @property
    def constant_data(self):
        cdef map[string, Blob.Ptr] c_constant_data
        c_constant_data = self.impl.constant_data
        constant_data = {}
        cdef BlobBuffer weights_buffer
        for weights in c_constant_data:
            weights_buffer = BlobBuffer()
            weights_buffer.reset(weights.second)
            constant_data[weights.first.decode()] = weights_buffer.to_numpy()
        return constant_data


cdef class Port:
    def __cinit__(self, shape: list=[]):
        cdef vector[size_t] c_shape
        for d in shape:
            c_shape.push_back(d)
        self.impl = C.Port(c_shape)
    @property
    def shape(self):
        return self.impl.shape

cdef class PortInfo:
    def __cinit__(self, layer_id: int = -1, port_id: int = -1):
        if layer_id != -1 and port_id != -1:
            self.impl = C.PortInfo(layer_id, port_id)
        else:
            self.impl = C.PortInfo()
    @property
    def layer_id(self):
        return self.impl.layer_id

    @property
    def port_id(self):
        return self.impl.port_id

    def __eq__(self, other):
        return self.layer_id == other.layer_id and self.port_id == other.port_id

    def __ne__(self, other):
        return self.layer_id != other.layer_id and self.port_id != other.port_id

cdef class Connection:
    def __cinit__(self, PortInfo input = None, PortInfo output = None):
        if input and output:
            self.impl = C.Connection(input.impl, output.impl)
        else:
            self.impl = C.Connection()
    @property
    def _from(self):
        cdef PortInfo port_info = PortInfo()
        port_info.impl = self.impl._from
        return port_info

    @property
    def to(self):
        cdef PortInfo port_info = PortInfo()
        port_info.impl = self.impl.to
        return port_info

    def __eq__(self, other):
        return self._from == other._from and self.to == other.to

    def __ne__(self, other):
        return self._from != other._from and self.to != other.to


def check_constant_data(data):
    for k, v in data.items():
        if not all([isinstance(x, type(v[0])) for x in v]):
            raise TypeError("Elements of list for key {} have different data types! "
                            "Please specify list of 'int' or 'float' values.".format(k))
        if isinstance(v, list):
            if isinstance(v[0], float):
                dtype = np.float32
            elif isinstance(v[0], int):
                dtype = np.int32
            else:
                raise TypeError("Unsupported precision of the data for key {}! Given {} but 'float  or 'int' precision expected".
                              format(k, str(v.dtype)))
            data[k] = np.asanyarray(v, dtype=dtype)
        elif isinstance(v, np.ndarray):
            pass
        else:
            raise TypeError("Unsupported data type for key '{}'. {} given but 'list' or 'numpy.ndarray' expected".
                            format(k, type(v)))
    return data


# TODO: Fix LAyerBuilder object copying - pass by reference
# cdef class LayerConstantData(dict):
#     def update(self, other=None, **kwargs):
#         if other:
#             other = check_constant_data(other)
#         cdef vector[size_t] dims
#         cdef Blob.Ptr blob_ptr
#         cdef BlobBuffer buffer
#         for k, v in other.items():
#             if k in self.keys() and (v.shape == self[k].shape and v.dtype == self[k].dtype):
#                 print("Reuse blob for {}\n".format(k))
#                 self[k][:] = v
#             else:
#                 for dim in v.shape:
#                     dims.push_back(dim)
#                 ie_precision = np_precision_map.get(str(v.dtype), None)
#                 if not ie_precision:
#                     raise BufferError("Unsupported precision of the data for key {}! Given {} but one of the {} precisions expected".
#                                       format(k, str(v.dtype), ", ".join(np_precision_map.keys())))
#                 blob_ptr = deref(self.impl).allocateBlob(dims, ie_precision.encode())
#                 buffer = BlobBuffer()
#                 buffer.reset(blob_ptr)
#                 np_buffer = buffer.to_numpy()
#                 np_buffer[:] = v
#                 deref(self.impl).addConstantData(k.encode(), blob_ptr)

cdef class LayerBuilder:

    def __cinit__(self, type: str=None, name: str=None):
        if name and type:
            self.impl = C.LayerBuilder(name.encode(), type.encode())
        else:
            self.impl = C.LayerBuilder()

    @property
    def id(self):
        return self.impl.id
    @property
    def name(self):
        return self.impl.getName().decode()
    @name.setter
    def name(self, name: str):
        self.impl.setName(name.encode())

    @property
    def type(self):
        return self.impl.getType().decode()
    @type.setter
    def type(self, type: str):
        self.impl.setType(type.encode())

    @property
    def input_ports(self):
        cdef Port port
        cdef vector[C.Port] c_ports = self.impl.getInputPorts()
        py_ports = []
        for p in c_ports:
            port = Port()
            port.impl = p
            py_ports.append(port)
        return py_ports

    @input_ports.setter
    def input_ports(self, ports: list):
        cdef vector[C.Port] c_ports
        cdef Port c_port
        for p in ports:
            c_port = Port(p.shape)
            c_ports.push_back(c_port.impl)
        self.impl.setInputPorts(c_ports)

    @property
    def output_ports(self):
        cdef Port port
        cdef vector[C.Port] c_ports = self.impl.getOutputPorts()
        py_ports = []
        for p in c_ports:
            port = Port()
            port.impl = p
            py_ports.append(port)
        return py_ports

    @output_ports.setter
    def output_ports(self, ports: list):
        cdef vector[C.Port] c_ports
        cdef Port c_port
        for p in ports:
            c_port = Port(p.shape)
            c_ports.push_back(c_port.impl)
        self.impl.setOutputPorts(c_ports)

    @property
    def params(self):
        return {k.decode(): v.decode() for k, v in self.impl.getParameters()}

    @params.setter
    def params(self, params_map: dict):
        cdef map[string, string] c_params_map
        for k, v in params_map.items():
            c_params_map[k.encode()] = str(v).encode()
        self.impl.setParameters(c_params_map)

    def build(self):
        cdef ILayer layer = ILayer()
        layer.impl = self.impl.build()
        return layer

    @property
    def constant_data(self):
        cdef map[string, Blob.Ptr] c_constant_data
        c_constant_data = self.impl.getConstantData()
        constant_data = {}
        # TODO: Fix LAyerBuilder object copying - pass by reference
        # constant_data = LayerConstantData()
        # constant_data.impl = make_shared[C.LayerBuilder](self.impl)
        cdef BlobBuffer weights_buffer
        for weights in c_constant_data:
            weights_buffer = BlobBuffer()
            weights_buffer.reset(weights.second)
            constant_data[weights.first.decode()] = weights_buffer.to_numpy()
        return constant_data

    @constant_data.setter
    def constant_data(self, data: dict):
        cdef vector[size_t] dims
        cdef map[string, Blob.Ptr] c_constant_data
        cdef Blob.Ptr blob_ptr
        cdef BlobBuffer buffer
        data = check_constant_data(data)
        for k, v in data.items():
            for dim in v.shape:
                dims.push_back(dim)
            ie_precision = np_precision_map.get(str(v.dtype), None)
            if not ie_precision:
                raise BufferError("Unsupported precision of the data for key {}! Given {} but one of the {} precisions expected".
                                  format(k, str(v.dtype), ", ".join(np_precision_map.keys())))
            blob_ptr = self.impl.allocateBlob(dims, ie_precision.encode())
            buffer = BlobBuffer()
            buffer.reset(blob_ptr)
            np_buffer = buffer.to_numpy()
            np_buffer[:] = v
            c_constant_data[k.encode()] = blob_ptr

        self.impl.setConstantData(c_constant_data)

    # TODO: Implement get\setGraph when will be supported