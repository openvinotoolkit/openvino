#distutils: language=c++
from cython.operator cimport dereference as deref
from .cimport ie_api_impl_defs as C
from .ie_api_impl_defs cimport Blob, TensorDesc, SizeVector, Precision
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr, shared_ptr
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint8_t
from libc.string cimport memcpy, strcpy
import os
import numpy as np
from copy import deepcopy
import warnings
from collections import OrderedDict, namedtuple
from collections import OrderedDict
import threading

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[C.IEExecNetwork] move(unique_ptr[C.IEExecNetwork])

cdef string to_std_string(str py_string):
    return py_string.encode()

cdef to_py_string(const string & std_string):
    return bytes(std_string).decode()

cdef dict_to_c_map(py_dict):
    cdef map[string, string] c_map
    for k, v in py_dict.items():
        if type(k) != str or type(v) != str:
            raise TypeError("Only string keys and values are allowed!")
        c_map[k.encode()] = v.encode()
    return c_map

cdef c_map_to_dict(map[string, string] c_map):
    py_dict = {}
    for v in c_map:
        py_dict[v.first.decode()] = v.second.decode()
    return py_dict

supported_precisions = ["FP32", "FP16", "Q78", "I32", "I16", "I8", "U32", "U16", "U8"]

supported_layouts = {0: "ANY", 1: "NCHW", 2: "NHWC", 3: "NCDHW", 4: "NDHWC", 64: "OIHW", 95: "SCALAR", 96: "C",
                     128: "CHW", 192: "HW", 193: "NC", 194: "CN", 200: "BLOCKED"}

known_plugins = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL', 'MULTI']

ctypedef enum StatusCode:
    OK = 0
    GENERAL_ERROR = -1
    NOT_IMPLEMENTED = -2
    NETWORK_NOT_LOADED = -3
    PARAMETER_MISMATCH = -4
    NOT_FOUND = -5
    OUT_OF_BOUNDS = -6
    UNEXPECTED = -7
    REQUEST_BUSY = -8
    RESULT_NOT_READY = -9
    NOT_ALLOCATED = -10
    INFER_NOT_STARTED = -11
    NETWORK_NOT_READ = -12

def get_version():
    return C.get_version().decode()

cdef class IECore:
    def __cinit__(self, xml_config_file: str = ""):
        self.impl = C.IECore(xml_config_file.encode())

    def get_versions(self, device_name: str):
        cdef  map[string, C.Version] versions_
        versions_ = self.impl.getVersions(device_name.encode())
        versions = {}
        for v in versions_:
            device = v.first.decode()
            ver = v.second
            versions[device] = namedtuple("Versions", ["major", "minor", "build_number", "description"])
            versions[device].build_number = ver.buildNumber.decode()
            versions[device].description = ver.description.decode()
            versions[device].minor = ver.apiVersion.minor
            versions[device].major = ver.apiVersion.major
        return versions

    cpdef ExecutableNetwork load_network(self, IENetwork network, str device_name, config=None, int num_requests=1):
        cdef ExecutableNetwork exec_net = ExecutableNetwork()
        cdef map[string, string] c_config

        if config:
            c_config = dict_to_c_map(config)
        exec_net.ie_core_impl = self.impl
        exec_net.impl = move(self.impl.loadNetwork(network.impl, device_name.encode(), c_config, num_requests))
        exec_net.inputs = network.inputs.keys()
        exec_net.outputs = list(network.outputs.keys())
        return exec_net

    def query_network(self, IENetwork network, str device_name, config=None):
        cdef map[string, string] c_config
        if config:
            c_config = dict_to_c_map(config)
        res = self.impl.queryNetwork(network.impl, device_name.encode(), c_config)
        return c_map_to_dict(res)

    def set_config(self, config: dict, device_name: str):
        cdef map[string, string] c_config = dict_to_c_map(config)
        self.impl.setConfig(c_config, device_name.encode())

    def register_plugin(self, plugin_name: str, device_name: str = ""):
        self.impl.registerPlugin(plugin_name.encode(), device_name.encode())

    def register_plugins(self, xml_config_file: str):
        self.impl.registerPlugins(xml_config_file.encode())

    def unregister_plugin(self, device_name: str):
        self.impl.unregisterPlugin(device_name.encode())

    def add_extension(self, extension_path: str, device_name: str):
        self.impl.addExtension(extension_path.encode(), device_name.encode())

    def get_metric(self, device_name: str, metric_name: str):
        return self.impl.getMetric(device_name.encode(), metric_name.encode())

    def get_config(self, device_name: str, config_name: str):
        return self.impl.getConfig(device_name.encode(), config_name.encode())

    @property
    def available_devices(self):
        cdef vector[string] c_devices = self.impl.getAvailableDevices()
        return [d.decode() for d in c_devices]

    # TODO: Add import network functionality
    # TODO: Extend API for query config and attributes when it will be merged in C++ API

cdef class DataPtr:
    @property
    def name(self):
        return deref(self._ptr).getName().decode()
    @property
    def precision(self):
        return deref(self._ptr).getPrecision().name().decode()
    @precision.setter
    def precision(self, precision):
        if precision not in supported_precisions:
            raise ValueError("Unsupported precision {}! List of supported precisions: {}".format(precision,
                                                                                                 supported_precisions))
        deref(self._ptr).setPrecision(C.Precision.FromStr(precision.encode()))

    @property
    def dims(self):
        return deref(self._ptr).getDims()
    @property
    def layout(self):
        return supported_layouts[deref(self._ptr).getLayout()]

    @property
    def initialized(self):
        return deref(self._ptr).isInitialized()

cdef class IENetLayer:
    @property
    def name(self):
        return self.impl.name.decode()
    @property
    def type(self):
        return self.impl.type.decode()
    @property
    def precision(self):
        warnings.filterwarnings("always", category=DeprecationWarning)
        warnings.warn("precision property of IENetLayer is deprecated. "
                      "Please use precision property of DataPtr instead",
                      DeprecationWarning)
        return self.impl.precision.decode()
    @property
    def affinity(self):
        return self.impl.affinity.decode()
    @property
    def weights(self):
        cdef map[string, Blob.Ptr] c_weights_map
        c_weights_map = self.impl.getWeights()
        weights_map = {}
        cdef BlobBuffer weights_buffer
        for weights in c_weights_map:
            weights_buffer = BlobBuffer()
            weights_buffer.reset(weights.second)
            weights_map[weights.first.decode()] = weights_buffer.to_numpy()
        return weights_map

    @property
    def params(self):
        return {k.decode(): v.decode() for k, v in self.impl.params}
    @property
    def parents(self):
        cdef vector[string] c_parents = self.impl.parents
        parents = []
        return [parent.decode() for parent in c_parents]
    @property
    def children(self):
        cdef vector[string] c_children = self.impl.children
        children = []
        return [child.decode() for child in c_children]
    @property
    def shape(self):
        string_shape = self.impl.shape.decode()
        return [int(i) for i in string_shape.split(' ')]
    @property
    def layout(self):
        warnings.filterwarnings("always", category=DeprecationWarning)
        warnings.warn("layout property of IENetLayer is deprecated. "
                      "Please use layout property of DataPtr instead",
                      DeprecationWarning)
        return self.impl.layout.decode()
    @affinity.setter
    def affinity(self, target_affinity):
        self.impl.setAffinity(target_affinity.encode())
    @params.setter
    def params(self, params_map):
        self.impl.setParams(dict_to_c_map(params_map))

    @precision.setter
    def precision(self, precision: str):
        self.impl.setPrecision(precision.upper().encode())

    @property
    def out_data(self):
        cdef vector[shared_ptr[C.Data]] out_data = self.impl.getOutData()
        data = []
        cdef DataPtr data_ptr = DataPtr()
        for d in out_data:
            data_ptr._ptr = d
            data.append(data_ptr)
        return data

cdef class InputInfo:
    @property
    def precision(self):
        return self.impl.precision.decode()
    @property
    def layout(self):
        return self.impl.layout.decode()
    @property
    def shape(self):
        return self.impl.dims

    @precision.setter
    def precision(self, precision):
        if precision.upper() not in supported_precisions:
            raise AttributeError(
                "Unsupported precision {}! List of supported precisions: {}".format(precision, supported_precisions))
        self.impl.setPrecision(precision.encode())
    @layout.setter
    def layout(self, layout):
        if layout.upper() not in supported_layouts.values():
            raise AttributeError(
                "Unsupported layout {}! List of supported layouts: {}".format(layout, supported_layouts.values()))
        self.impl.setLayout(layout.encode())

cdef class OutputInfo:
    @property
    def precision(self):
        return self.impl.precision.decode()
    @property
    def layout(self):
        return self.impl.layout.decode()
    @property
    def shape(self):
        return self.impl.dims
    @precision.setter
    def precision(self, precision):
        if precision.upper() not in supported_precisions:
            raise AttributeError(
                "Unsupported precision {}! List of supported precisions: {}".format(precision, supported_precisions))
        self.impl.setPrecision(precision.encode())

cdef class ExecutableNetwork:
    def __init__(self):
        self._infer_requests = []
        self._requests = []
        self.inputs = []
        self.outputs = []

    def infer(self, inputs=None):
        current_request = self.requests[0]
        current_request.infer(inputs)
        return deepcopy(current_request.outputs)

    def start_async(self, request_id, inputs=None):
        if request_id not in list(range(len(self.requests))):
            raise ValueError("Incorrect request_id specified!")
        current_request = self.requests[request_id]
        current_request.async_infer(inputs)
        return current_request

    @property
    def requests(self):
        if (len(self._infer_requests) == 0):
            for i in range(deref(self.impl).infer_requests.size()):
                infer_request = InferRequest()
                infer_request.impl = &(deref(self.impl).infer_requests[i])
                self._infer_requests.append(infer_request)

        if (len(self._infer_requests) != deref(self.impl).infer_requests.size()):
            raise Exception("Mismatch of infer requests number!")

        for i in range(len(self._infer_requests)):
            self._infer_requests[i]._inputs_list = self.inputs
            self._infer_requests[i]._outputs_list = self.outputs

        return self._infer_requests

    def get_exec_graph_info(self):
        ie_network = IENetwork()
        ie_network.impl = deref(self.impl).GetExecGraphInfo()
        return ie_network

    def get_metric(self, metric_name: str):
        return deref(self.impl).getMetric(metric_name.encode())

    def get_config(self, config_name: str):
        return deref(self.impl).getConfig(config_name.encode())

ctypedef extern void (*cb_type)(void*, int) with gil

cdef class InferRequest:
    def __init__(self):
        self._inputs_list = []
        self._outputs_list = []
        self._py_callback = lambda *args, **kwargs: None
        self._py_callback_used = False
        self._py_callback_called = threading.Event()
        self._py_data = None

    cdef void user_callback(self, int status) with gil:
        if self._py_callback:
            self._py_callback(status, self._py_data)
            self._py_callback_called.set()

    def set_completion_callback(self, py_callback, py_data = None):
        self._py_callback = py_callback
        self._py_data = py_data
        self._py_callback_used = True
        deref(self.impl).setCyCallback(<cb_type> self.user_callback, <void *> self)

    cpdef BlobBuffer _get_blob_buffer(self, const string & blob_name):
        cdef BlobBuffer buffer = BlobBuffer()
        cdef Blob.Ptr blob_ptr
        deref(self.impl).getBlobPtr(blob_name, blob_ptr)
        buffer.reset(blob_ptr)
        return buffer

    cpdef infer(self, inputs=None):
        if inputs is not None:
            self._fill_inputs(inputs)

        deref(self.impl).infer()

    cpdef async_infer(self, inputs=None):
        if inputs is not None:
            self._fill_inputs(inputs)
        self._py_callback_called.clear()
        deref(self.impl).infer_async()

    cpdef wait(self, timeout=None):
        if self._py_callback_used:
            while not self._py_callback_called.is_set():
                if not self._py_callback_called.wait(timeout):
                    return StatusCode.REQUEST_BUSY
            return StatusCode.OK
        else:
            if timeout is None:
                timeout = -1
            return deref(self.impl).wait(<int64_t> timeout)

    cpdef get_perf_counts(self):
        cdef map[string, C.ProfileInfo] c_profile = deref(self.impl).getPerformanceCounts()
        profile = {}
        for l in c_profile:
            info = l.second
            # TODO: add execution index. Check if unsigned int is properly converted to int in python.
            profile[l.first.decode()] = {"status": info.status.decode(), "exec_type": info.exec_type.decode(),
                                         "layer_type": info.layer_type.decode(), "real_time": info.real_time,
                                         "cpu_time": info.cpu_time, "execution_index": info.execution_index}
        return profile

    @property
    def inputs(self):
        inputs = {}
        for input in self._inputs_list:
            inputs[input] = self._get_blob_buffer(input.encode()).to_numpy()
        return inputs

    @property
    def outputs(self):
        outputs = {}
        for output in self._outputs_list:
            outputs[output] = self._get_blob_buffer(output.encode()).to_numpy()
        return deepcopy(outputs)

    @property
    def latency(self):
        return self.impl.exec_time

    def set_batch(self, size):
        if size <= 0:
            raise ValueError("Batch size should be positive integer number but {} specified".format(size))
        deref(self.impl).setBatch(size)

    def _fill_inputs(self, inputs):
        for k, v in inputs.items():
            assert k in self._inputs_list, "No input with name {} found in network".format(k)
            self.inputs[k][:] = v


class LayerStats:
    def __init__(self, min: tuple = (), max: tuple = ()):
        self._min = min
        self._max = max

    @property
    def min(self):
        return self._min
    @property
    def max(self):
        return self._max


cdef class LayersStatsMap(dict):
    def update(self, other=None, **kwargs):
        super(LayersStatsMap, self).update(other, **kwargs)
        cdef map[string, map[string, vector[float]]] c_stats_map
        cdef map[string, vector[float]] c_node_stats
        for k, v in self.items():
            c_node_stats["min".encode()] = v.min
            c_node_stats["max".encode()] = v.max
            c_stats_map[k.encode()] = c_node_stats
        self.net_impl.setStats(c_stats_map)

cdef class IENetwork:
    def __cinit__(self, model: [str, bytes] = "", weights: [str, bytes] = "", init_from_buffer: bool = False):
        cdef char*xml_buffer = <char*> malloc(len(model))
        cdef uint8_t*bin_buffer = <uint8_t *> malloc(len(weights))
        cdef string model_
        cdef string weights_
        if init_from_buffer:
            strcpy(xml_buffer, model)
            memcpy(bin_buffer, <uint8_t *> weights, len(weights))
            self.impl = C.IENetwork()
            self.impl.load_from_buffer(xml_buffer, len(model), bin_buffer, len(weights))
        else:
            if model and weights:
                if not os.path.isfile(model):
                    raise Exception("Path to the model {} doesn't exists or it's a directory".format(model))
                if not os.path.isfile(weights):
                    raise Exception("Path to the weights {} doesn't exists or it's a directory".format(weights))
                model_ = model.encode()
                weights_ = weights.encode()
                self.impl = C.IENetwork(model_, weights_)
            else:
                self.impl = C.IENetwork()
        free(xml_buffer)

    @property
    def name(self):
        name = bytes(self.impl.name)
        return name.decode()

    @property
    def inputs(self):
        cdef map[string, C.InputInfo] c_inputs = self.impl.getInputs()
        inputs = {}
        cdef InputInfo in_info
        for input in c_inputs:
            in_info = InputInfo()
            in_info.impl = input.second
            inputs[input.first.decode()] = in_info
        return inputs

    @property
    def outputs(self):
        cdef map[string, C.OutputInfo] c_outputs = self.impl.getOutputs()
        outputs = {}
        cdef OutputInfo out_info
        for out in c_outputs:
            out_info = OutputInfo()
            out_info.impl = out.second
            outputs[out.first.decode()] = out_info
        return outputs

    @property
    def batch_size(self):
        return self.impl.batch_size

    @property
    def precision(self):
        return self.impl.precision.decode()

    @batch_size.setter
    def batch_size(self, batch: int):
        if batch <= 0:
            raise AttributeError("Invalid batch size {}! Batch size should be positive integer value".format(batch))
        self.impl.setBatch(batch)
        self.impl.batch_size = batch

    @property
    def layers(self):
        cdef vector[pair[string, C.IENetLayer]] c_layers = self.impl.getLayers()
        layers = OrderedDict()
        cdef IENetLayer net_l = IENetLayer()
        for l in c_layers:
            net_l = IENetLayer()
            net_l.impl = l.second
            layers[l.first.decode()] = net_l
        return layers
    @property
    def stats(self):
        cdef map[string, map[string, vector[float]]] c_stats_map = self.impl.getStats()
        py_stats_map = LayersStatsMap()
        py_stats_map.net_impl = self.impl
        for it in c_stats_map:
            stats_map = LayersStatsMap()
            py_stats_map[it.first.decode()] = LayerStats(min=tuple(it.second["min".encode()]),
                                                         max=tuple(it.second["max".encode()]))
        return py_stats_map

    @classmethod
    def from_ir(cls, model: str, weights: str):
        warnings.filterwarnings("always", category=DeprecationWarning)
        warnings.warn("from_ir() method of IENetwork is deprecated. "
                      "Please use IENetwork class constructor to create valid IENetwork instance",
                      DeprecationWarning)
        if not os.path.isfile(model):
            raise Exception("Path to the model {} doesn't exists or it's a directory".format(model))
        if not os.path.isfile(weights):
            raise Exception("Path to the weights {} doesn't exists or it's a directory".format(weights))
        cdef IENetwork net = IENetwork(model, weights)
        return net

    def add_outputs(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        for i, l in enumerate(outputs):
            if isinstance(l, str):
                self.impl.addOutput(l.encode(), 0)
            elif isinstance(l, tuple) and len(l) == 2:
                self.impl.addOutput(l[0].encode(), l[1])
            else:
                raise TypeError("Incorrect type {type} for layer to add at index {ind}. "
                                "Expected string with layer name or tuple with two elements: layer name as "
                                "first element and port id as second".format(type=type(l), ind=i))

    def serialize(self, path_to_xml, path_to_bin: str = ""):
        self.impl.serialize(path_to_xml.encode(), path_to_bin.encode())

    def reshape(self, input_shapes: dict):
        cdef map[string, vector[size_t]] c_input_shapes;
        cdef vector[size_t] c_shape
        net_inputs = self.inputs
        for input, shape in input_shapes.items():
            c_shape = []
            if input not in net_inputs:
                raise AttributeError("Specified '{}' layer not in network inputs '{}'! ".format(input, net_inputs))
            for v in shape:
                c_shape.push_back(v)
            c_input_shapes[input.encode()] = c_shape
        self.impl.reshape(c_input_shapes)

cdef class IEPlugin:
    def __cinit__(self, device: str, plugin_dirs=None):
        plugin_base = device.split(':')[0]
        if plugin_base not in known_plugins:
            raise ValueError("Unknown plugin: {}, expected one of: {}"
                             .format(plugin_base, ",".join(known_plugins)))
        if plugin_dirs is None:
            plugin_dirs = [""]
        elif isinstance(plugin_dirs, str):
            plugin_dirs = [plugin_dirs]

        # add package directory to plugin_dirs
        lib_location = os.path.dirname(os.path.realpath(__file__))
        plugin_dirs.append(lib_location)

        cpdef string device_ = <string> device.encode()
        cdef vector[string] dirs_
        for d in plugin_dirs:
            dirs_.push_back(<string> d.encode())

        self.impl = C.IEPlugin(device_, dirs_)

    cpdef ExecutableNetwork load(self, IENetwork network, int num_requests=1, config=None):
        cdef ExecutableNetwork exec_net = ExecutableNetwork()
        cdef map[string, string] c_config
        if num_requests < 0:
            raise ValueError("Incorrect number of requests specified: {}. Expected positive integer number "
                             "or zero for auto detection".format(num_requests))
        if config:
            for k, v in config.items():
                c_config[to_std_string(k)] = to_std_string(v)
        exec_net.plugin_impl = self.impl
        exec_net.impl = move(self.impl.load(network.impl, num_requests, c_config))
        exec_net.inputs = network.inputs.keys()
        exec_net.outputs = list(network.outputs.keys())
        return exec_net

    cpdef void set_initial_affinity(self, IENetwork net) except *:
        if self.device.find("HETERO") == -1:
            raise RuntimeError("set_initial_affinity method applicable only for HETERO device")
        self.impl.setInitialAffinity(net.impl)

    cpdef set get_supported_layers(self, IENetwork net):
        return set([l.decode() for l in self.impl.queryNetwork(net.impl)])

    @property
    def device(self):
        device_name = bytes(self.impl.device_name)
        return to_py_string(device_name)

    @property
    def version(self):
        version = bytes(self.impl.version)
        return version.decode()

    cpdef void add_cpu_extension(self, str extension_path) except *:
        if self.device.find("CPU") == -1:
            raise RuntimeError("add_cpu_extension method applicable only for CPU or HETERO devices")
        cdef string extension_str = extension_path.encode()
        self.impl.addCpuExtension(extension_str)

    cpdef void set_config(self, config):
        cdef map[string, string] c_config
        for k, v in config.items():
            c_config[to_std_string(k)] = to_std_string(v)
        self.impl.setConfig(c_config)

    # TODO: Add export compiled network functionality

cdef class BlobBuffer:
    """Copy-less accessor for Inference Engine Blob"""

    cdef reset(self, Blob.Ptr & ptr):
        self.ptr = ptr
        cdef TensorDesc desc = deref(ptr).getTensorDesc()
        cdef SizeVector shape = desc.getDims()
        cdef Py_ssize_t itemsize = deref(ptr).element_size()
        self.strides.resize(shape.size())
        self.shape.resize(shape.size())

        total_stride = itemsize
        # dims are in row major (C - style),
        # thence strides are computed starting from latest dimension
        for i in reversed(range(shape.size())):
            self.strides[i] = total_stride
            self.shape[i] = shape[i]
            total_stride *= shape[i]

        self.total_stride = total_stride
        self.format = self._get_blob_format(desc)
        self.item_size = itemsize

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = C.get_buffer[char](deref(self.ptr))
        buffer.format = self.format
        buffer.internal = NULL
        buffer.itemsize = self.item_size
        buffer.len = self.total_stride
        buffer.ndim = self.shape.size()
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape.data()
        buffer.strides = self.strides.data()
        buffer.suboffsets = NULL

    cdef char*_get_blob_format(self, const TensorDesc & desc):
        cdef Precision precision = desc.getPrecision()
        name = bytes(precision.name()).decode()
        # todo: half floats
        precision_to_format = {
            'FP32': 'f',  # float
            'FP16': 'h',  # signed short
            'Q78': 'h',  # signed short
            'I16': 'h',  # signed short
            'U8': 'B',  # unsigned char
            'I8': 'b',  # signed char
            'U16': 'H',  # unsigned short
            'I32': 'i'  # signed int
        }

        if name not in precision_to_format:
            raise ValueError("Unknown Blob precision: {}".format(name))

        return precision_to_format[name].encode()

    def to_numpy(self):
        return np.asarray(self)
