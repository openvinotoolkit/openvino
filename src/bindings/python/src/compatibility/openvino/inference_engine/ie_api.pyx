# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#distutils: language=c++
#cython: embedsignature=True
#cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint8_t, int8_t, int32_t, uint16_t, int16_t, uint32_t, uint64_t
from libc.stddef cimport size_t
from libc.string cimport memcpy

import os
from fnmatch import fnmatch
import threading
import warnings
from copy import deepcopy
from collections import namedtuple

from .cimport ie_api_impl_defs as C
from .ie_api_impl_defs cimport SizeVector, Precision
from .constants import WaitMode, StatusCode, MeanVariant, layout_str_to_enum, format_map, layout_int_to_str_map,\
    known_plugins, supported_precisions, ResizeAlgorithm, ColorFormat

import numpy as np

warnings.filterwarnings(action="module", category=DeprecationWarning)

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[C.IEExecNetwork] move(unique_ptr[C.IEExecNetwork])

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


def get_version():
    return C.get_version().decode()


def read_network(path_to_xml : str, path_to_bin : str):
    cdef IENetwork net = IENetwork()
    net.impl = C.read_network(path_to_xml.encode(), path_to_bin.encode())
    return net


cdef class VariableState:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class manages data for reset operations
    """

    def reset(self):
        """
        Reset internal variable state for relevant infer request
        to a value specified as default for according ReadValue node
        """
        self.impl.reset()


    @property
    def state(self):
        """
        Returns the value of the variable state.
        """
        blob = Blob()
        blob._ptr = self.impl.getState()
        blob._is_const = True
        return blob

    @state.setter
    def state(self, blob : Blob):
        self.impl.setState(blob._ptr)

    @property
    def name(self):
        """
        A string representing a state name
        """
        return to_py_string(self.impl.getName())


cdef class TensorDesc:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class defines Tensor description
    """

    def __eq__(self, other : TensorDesc):
        return self.layout == other.layout and self.precision == other.precision and self.dims == other.dims

    def __ne__(self, other : TensorDesc):
        return self.layout != other.layout or self.precision != other.precision or self.dims != other.dims

    def __deepcopy__(self, memodict={}):
        return TensorDesc(deepcopy(self.precision, memodict), deepcopy(self.dims, memodict), deepcopy(self.layout, memodict))


    def __cinit__(self, precision : str, dims : [list, tuple], layout : str):
        """Class constructor

        :param precision: target memory precision
        :param dims: target memory dimensions
        :param layout: target memory layout
        :return: Instance of defines class
        """
        if precision not in supported_precisions:
            raise ValueError(f"Unsupported precision {precision}! List of supported precisions: {supported_precisions}")
        self.impl = C.CTensorDesc(C.Precision.FromStr(precision.encode()), dims, layout_str_to_enum[layout])


    @property
    def dims(self):
        """
        Shape (dimensions) of the :class:`TensorDesc` object
        """
        return self.impl.getDims()

    @dims.setter
    def dims(self, dims_array : [list, tuple]):
        self.impl.setDims(dims_array)


    @property
    def precision(self):
        """
        Precision of the :class:`TensorDesc` object
        """
        return self.impl.getPrecision().name().decode()

    @precision.setter
    def precision(self, precision : str):
        if precision not in supported_precisions:
            raise ValueError(f"Unsupported precision {precision}! List of supported precisions: {supported_precisions}")
        self.impl.setPrecision(C.Precision.FromStr(precision.encode()))


    @property
    def layout(self):
        """
        Layout of the :class:`TensorDesc` object
        """
        return layout_int_to_str_map[self.impl.getLayout()]

    @layout.setter
    def layout(self, layout : str):
        if layout not in layout_str_to_enum.keys():
            raise ValueError(f"Unsupported layout {layout}! "
                             f"List of supported layouts: {list(layout_str_to_enum.keys())}")
        self.impl.setLayout(layout_str_to_enum[layout])


cdef class Blob:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class represents Blob
    """

    def __cinit__(self, TensorDesc tensor_desc = None, array : np.ndarray = None):
        """Class constructor

        :param tensor_desc: :class:`TensorDesc` object describing creating Blob object.
        :param array: numpy.ndarray with data to fill blob memory, The array have to have same elements count
        as specified in tensor_desc.dims attribute and same elements precision corresponding to
        tensor_desc.precision. If array isn't provided empty numpy.ndarray will be created accorsing
        to parameters of tensor_desc
        :return: Instance of Blob class
        """
        cdef CTensorDesc c_tensor_desc
        cdef float[::1] fp32_array_memview
        cdef double[::1] fp64_array_memview
        cdef int16_t[::1] I16_array_memview
        cdef uint16_t[::1] U16_array_memview
        cdef uint8_t[::1] U8_array_memview
        cdef int8_t[::1] I8_array_memview
        cdef int32_t[::1] I32_array_memview
        cdef int64_t[::1] I64_array_memview
        cdef uint32_t[::1] U32_array_memview
        cdef uint64_t[::1] U64_array_memview

        self._is_const = False
        self._array_data = array
        self._initial_shape = array.shape if array is not None else None

        if self._array_data is not None:
            if np.isfortran(self._array_data):
                self._array_data = self._array_data.ravel(order="F")
            else:
                self._array_data = self._array_data.ravel(order="C")
        if self._array_data is None and tensor_desc is not None:
            c_tensor_desc = tensor_desc.impl
            precision = tensor_desc.precision
            if precision == "FP32":
                self._ptr = C.make_shared_blob[float](c_tensor_desc)
            elif precision == "FP64":
                self._ptr = C.make_shared_blob[double](c_tensor_desc)
            elif precision == "FP16" or precision == "I16" or precision == "BF16":
                self._ptr = C.make_shared_blob[int16_t](c_tensor_desc)
            elif precision == "Q78" or precision == "U16":
                self._ptr = C.make_shared_blob[uint16_t](c_tensor_desc)
            elif precision == "U8" or precision == "BOOL":
                self._ptr = C.make_shared_blob[uint8_t](c_tensor_desc)
            elif precision == "I8" or precision == "BIN" or precision == "I4" or precision == "U4":
                self._ptr = C.make_shared_blob[int8_t](c_tensor_desc)
            elif precision == "I32":
                self._ptr = C.make_shared_blob[int32_t](c_tensor_desc)
            elif precision == "U32":
                self._ptr = C.make_shared_blob[uint32_t](c_tensor_desc)
            elif precision == "I64":
                self._ptr = C.make_shared_blob[int64_t](c_tensor_desc)
            elif precision == "U64":
                self._ptr = C.make_shared_blob[uint64_t](c_tensor_desc)
            else:
                raise AttributeError(f"Unsupported precision {precision} for blob")
            deref(self._ptr).allocate()
        elif tensor_desc is not None and self._array_data is not None:
            c_tensor_desc = tensor_desc.impl
            precision = tensor_desc.precision
            size_td = C.product(c_tensor_desc.getDims())
            if array.size != size_td:
                raise AttributeError(f"Number of elements in provided numpy array {array.size} and "
                                     f"required by TensorDesc {size_td} are not equal")
            if self._array_data.dtype != format_map[precision]:
                raise ValueError(f"Data type {self._array_data.dtype} of provided numpy array "
                                 f"doesn't match to TensorDesc precision {precision}")
            if not self._array_data.flags['C_CONTIGUOUS']:
                self._array_data = np.ascontiguousarray(self._array_data)
            if precision == "FP32":
                fp32_array_memview = self._array_data
                self._ptr = C.make_shared_blob[float](c_tensor_desc, &fp32_array_memview[0], fp32_array_memview.shape[0])
            elif precision == "FP64":
                fp64_array_memview = self._array_data
                self._ptr = C.make_shared_blob[double](c_tensor_desc, &fp64_array_memview[0], fp64_array_memview.shape[0])
            elif precision == "FP16" or precision == "BF16":
                I16_array_memview = self._array_data.view(dtype=np.int16)
                self._ptr = C.make_shared_blob[int16_t](c_tensor_desc, &I16_array_memview[0], I16_array_memview.shape[0])
            elif precision == "I16":
                I16_array_memview = self._array_data
                self._ptr = C.make_shared_blob[int16_t](c_tensor_desc, &I16_array_memview[0], I16_array_memview.shape[0])
            elif precision == "Q78" or precision == "U16":
                U16_array_memview = self._array_data
                self._ptr = C.make_shared_blob[uint16_t](c_tensor_desc, &U16_array_memview[0], U16_array_memview.shape[0])
            elif precision == "U8" or precision == "BOOL":
                U8_array_memview = self._array_data
                self._ptr = C.make_shared_blob[uint8_t](c_tensor_desc, &U8_array_memview[0], U8_array_memview.shape[0])
            elif precision == "I8" or precision == "BIN" or precision == "I4" or precision == "U4":
                I8_array_memview = self._array_data
                self._ptr = C.make_shared_blob[int8_t](c_tensor_desc, &I8_array_memview[0], I8_array_memview.shape[0])
            elif precision == "I32":
                I32_array_memview = self._array_data
                self._ptr = C.make_shared_blob[int32_t](c_tensor_desc, &I32_array_memview[0], I32_array_memview.shape[0])
            elif precision == "U32":
                U32_array_memview = self._array_data
                self._ptr = C.make_shared_blob[uint32_t](c_tensor_desc, &U32_array_memview[0], U32_array_memview.shape[0])
            elif precision == "I64":
                I64_array_memview = self._array_data
                self._ptr = C.make_shared_blob[int64_t](c_tensor_desc, &I64_array_memview[0], I64_array_memview.shape[0])
            elif precision == "U64":
                U64_array_memview = self._array_data
                self._ptr = C.make_shared_blob[uint64_t](c_tensor_desc, &U64_array_memview[0], U64_array_memview.shape[0])
            else:
                raise AttributeError(f"Unsupported precision {precision} for blob")

    def __deepcopy__(self, memodict):
        res = Blob(deepcopy(self.tensor_desc, memodict), deepcopy(self._array_data, memodict))
        res.buffer[:] = deepcopy(self.buffer[:], memodict)
        return res

    @property
    def buffer(self):
        """
        Blob's memory as :class:`numpy.ndarray` representation
        """
        representation_shape = self._initial_shape if self._initial_shape is not None else []
        cdef BlobBuffer buffer = BlobBuffer()
        buffer.reset(self._ptr, representation_shape)
        return buffer.to_numpy(self._is_const)


    @property
    def tensor_desc(self):
        """
        :class:`TensorDesc` of created Blob
        """
        cdef CTensorDesc c_tensor_desc = deref(self._ptr).getTensorDesc()
        precision = c_tensor_desc.getPrecision().name().decode()
        layout = c_tensor_desc.getLayout()
        dims = c_tensor_desc.getDims()
        tensor_desc = TensorDesc(precision, dims, layout_int_to_str_map[layout])
        return tensor_desc

    def set_shape(self, new_shape):
        self._initial_shape = new_shape
        deref(self._ptr).setShape(new_shape)


## This class represents an Inference Engine entity and allows you to manipulate with plugins using unified interfaces.
cdef class IECore:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class represents an Inference Engine entity and allows you to manipulate with plugins using unified interfaces.
    """

    def __cinit__(self, xml_config_file: str = ""):
        """Class constructor

        :param xml_config_file:  A full path to `.xml` file containing plugins configuration.
        If the parameter is not specified, the default configuration is handled automatically.
        :return: Instance of IECore class
        """
        cdef string c_xml_config_file = xml_config_file.encode()
        with nogil:
            self.impl = C.IECore(c_xml_config_file)


    def get_versions(self, device_name: str):
        """Get a :class:`collections.namedtuple` object with versions of the plugin specified
        
        :param device_name: Name of the the registered plugin
        :return: Dictionary mapping a plugin name and `Versions` :class:`collections.namedtuple` object with the following fields:
        
        * `major` - major plugin integer version
        * `minor` - minor plugin integer version
        * `build_number` - plugin build number string
        * `description` - plugin description string
        """
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

    ## Reads a network from Intermediate Representation (IR) or ONNX formats and creates an `IENetwork`.
    #  @param model: A `.xml` or `.onnx` model file or string with IR.
    #  @param weights: A `.bin` file of the IR. Depending on `init_from_buffer` value, can be a string path or
    #                  bytes with file content.
    #  @param init_from_buffer: Defines the way of how `model` and `weights` attributes are interpreted.
    #                           If  `False`, attributes are interpreted as strings with paths to .xml and .bin files
    #                           of IR. If `True`, they are  interpreted as Python `bytes` object with .xml and .bin files content.
    #  @return An `IENetwork` object
    #
    #  Usage example:\n
    #  ```python
    #  ie = IECore()
    #  net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
    #  ```
    cpdef IENetwork read_network(self, model: [str, bytes, os.PathLike], weights: [str, bytes, os.PathLike] = "", init_from_buffer: bool = False):
        """Reads a network from Intermediate Representation (IR) or ONNX formats and creates an :class:`IENetwork`.
        
        :param model: A `.xml`, `.onnx`or `.prototxt` model file or string with IR.
        :param weights: A `.bin` file of the IR. Depending on `init_from_buffer` value, can be a string path or
                        bytes with file content.
        :param init_from_buffer: Defines the way of how `model` and `weights` attributes are interpreted.
        If  `False`, attributes are interpreted as strings with paths to `.xml` and `.bin` files
        of IR. If `True`, they are  interpreted as Python `bytes` object with `.xml` and `.bin` files content.
        
        :return: An :class:`IENetwork` object
        
        Usage example:
        
        .. code-block:: python
        
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
        """
        cdef uint8_t*bin_buffer
        cdef string weights_
        cdef string model_
        cdef IENetwork net = IENetwork()
        cdef size_t bin_size
        if init_from_buffer:
            model_ = bytes(model)
            bin_buffer = <uint8_t*> weights
            bin_size = len(weights)
            with nogil:
                net.impl = self.impl.readNetwork(model_, bin_buffer, bin_size)
        else:
            weights_ = "".encode()
            model = os.fspath(model)
            if not os.path.isfile(model):
                raise Exception(f"Path to the model {model} doesn't exist or it's a directory")
            model_ = model.encode()

            if not (fnmatch(model, "*.onnx") or fnmatch(model, "*.prototxt")) and weights:
                weights = os.fspath(weights)
                if not os.path.isfile(weights):
                    raise Exception(f"Path to the weights {weights} doesn't exist or it's a directory")
                weights_ = weights.encode()
            with nogil:
                net.impl = self.impl.readNetwork(model_, weights_)
        return net

    cpdef ExecutableNetwork load_network(self, network: [IENetwork, str], device_name=None, config=None, int num_requests=1):
        """Loads a network that was read from the Intermediate Representation (IR) to the plugin with specified device name
        and creates an :class:`ExecutableNetwork` object of the :class:`IENetwork` class.
        You can create as many networks as you need and use them simultaneously (up to the limitation of the hardware
        resources).
        
        :param network: A valid :class:`IENetwork` instance. Model file name .xml, .onnx can also be passed as argument
        :param device_name: A device name of a target plugin, if no device_name is set then it will use AUTO device as default.
        :param config: A dictionary of plugin configuration keys and their values
        :param num_requests: A positive integer value of infer requests to be created.
        Number of infer requests is limited by device capabilities. Value `0` indicates that optimal number of infer requests will be created.
        
        :return: An :class:`ExecutableNetwork` object
        
        Usage example:
        
        .. code-block:: python

            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
        """
        cdef ExecutableNetwork exec_net = ExecutableNetwork()
        cdef map[string, string] c_config
        cdef string c_device_name
        cdef string c_network_path
        if num_requests < 0:
            raise ValueError(f"Incorrect number of requests specified: {num_requests}. Expected positive integer number "
                             "or zero for auto detection")
        if config:
            c_config = dict_to_c_map(config)
        if device_name:
            c_device_name = device_name.encode()
        if isinstance(network, str):
            c_network_path = network.encode()
            if device_name:
                with nogil:
                    exec_net.impl = move(self.impl.loadNetworkFromFile(c_network_path, c_device_name, c_config, num_requests))
            else:
                with nogil:
                    exec_net.impl = move(self.impl.loadNetworkFromFile(c_network_path, c_config, num_requests))
        else:
            if device_name:
                with nogil:
                    exec_net.impl = move(self.impl.loadNetwork((<IENetwork>network).impl, c_device_name, c_config, num_requests))
            else:
                with nogil:
                    exec_net.impl = move(self.impl.loadNetwork((<IENetwork>network).impl, c_config, num_requests))
        return exec_net

    cpdef ExecutableNetwork import_network(self, str model_file, str device_name, config=None, int num_requests=1):
        """Creates an executable network from a previously exported network
        
        :param device_name: Name of device load executable network on
        :param model_file: Full path to the location of the exported file
        :param config: A dictionary of plugin configuration keys and their values
        :param num_requests: A positive integer value of infer requests to be created. Number of infer requests is limited
                             by device capabilities.
                             Value `0` indicates that optimal number of infer requests will be created.
        
        :return: An :class:`ExecutableNetwork` object
        
        Usage example:
        
        .. code-block:: python
        
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
            # export executable network
            exec_net.export(path_to_file_to_save)
            # import previously exported executable network
            exec_net_imported = ie.import_network(model_file=path_to_file_to_save, device_name="CPU")
        """
        cdef ExecutableNetwork exec_net = ExecutableNetwork()
        cdef map[string, string] c_config
        if num_requests < 0:
            raise ValueError(f"Incorrect number of requests specified: {num_requests}. Expected positive integer number "
                             "or zero for auto detection")
        if config:
            c_config = dict_to_c_map(config)
        exec_net.impl = move(self.impl.importNetwork(model_file.encode(), device_name.encode(), c_config, num_requests))
        return exec_net


    def query_network(self, IENetwork network, str device_name, config=None):
        """Queries the plugin with specified device name what network layers are supported in the current configuration.
        Please note that layers support depends on plugin configuration and loaded extensions.
        
        :param network: A valid :class:`IENetwork` instance
        :param device_name: A device name of a target plugin
        :param config: A dictionary of plugin configuration keys and their values
        :return: A dictionary mapping layers and device names on which they are supported
    
        Usage example:
    
        .. code-block:: python
        
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            layers_map = ie.query_network(network=net, device_name="HETERO:GPU,CPU")
        """
        cdef map[string, string] c_config
        if config:
            c_config = dict_to_c_map(config)
        res = self.impl.queryNetwork(network.impl, device_name.encode(), c_config)
        return c_map_to_dict(res)


    def set_config(self, config: dict, device_name: str):
        """Sets a configuration for a plugin
        
        .. note:: When specifying a key value of a config, the "KEY_" prefix is omitted.
        
        :param config: a dictionary of configuration parameters as keys and their values
        :param device_name: a device name of a target plugin
        :return: None
        
        Usage examples:\n
        
        .. code-block:: python
        
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            ie.set_config(config={"PERF_COUNT": "YES"}, device_name="CPU")
        """
        cdef map[string, string] c_config = dict_to_c_map(config)
        self.impl.setConfig(c_config, device_name.encode())


    def register_plugin(self, plugin_name: str, device_name: str = ""):
        """Register a new device and plugin that enables this device inside OpenVINO Runtime.
        
        :param plugin_name: A path (absolute or relative) or name of a plugin. Depending on platform, 
                            `plugin` is wrapped with shared library suffix and prefix to identify library full name
        :param device_name: A target device name for the plugin. If not specified, the method registers
                            a plugin with the default name.
        
        :return: None
    
        Usage example:

        .. code-block:: python

            ie = IECore()
            ie.register_plugin(plugin_name="openvino_intel_cpu_plugin", device_name="MY_NEW_PLUGIN")
        """
        self.impl.registerPlugin(plugin_name.encode(), device_name.encode())


    def register_plugins(self, xml_config_file: str):
        """Registers plugins specified in an `.xml` configuration file
        
        :param xml_config_file: A full path to `.xml` file containing plugins configuration
        :return: None

        Usage example:

        .. code-block:: python

            ie = IECore()
            ie.register_plugins("/localdisk/plugins/my_custom_cfg.xml")
        """
        self.impl.registerPlugins(xml_config_file.encode())

   
    def unregister_plugin(self, device_name: str):
        """Unregisters a plugin with a specified device name
        
        :param device_name: A device name of the plugin to unregister
        :return: None
        
        Usage example:

        .. code-block:: python

            ie = IECore()
            ie.unregister_plugin(device_name="GPU")
        """
        self.impl.unregisterPlugin(device_name.encode())

    
    def add_extension(self, extension_path: str, device_name: str):
        """Loads extension library to the plugin with a specified device name
        
        :param extension_path: Path to the extensions library file to load to a plugin
        :param device_name: A device name of a plugin to load the extensions to
        :return: None

        Usage example:\n
        
        .. code-block:: python
        
             ie = IECore()
             ie.add_extension(extension_path="/some_dir/libcpu_extension_avx2.so", device_name="CPU")
        """
        self.impl.addExtension(extension_path.encode(), device_name.encode())

    def get_metric(self, device_name: str, metric_name: str):
        """
        Gets a general runtime metric for dedicated hardware. Enables to request common device properties,
        which are :class:`ExecutableNetwork` agnostic, such as device name, temperature, and other devices-specific values.
        
        :param device_name: A name of a device to get a metric value.
        :param metric_name: A metric name to request.
        :return: A metric value corresponding to a metric key.

        Usage example:

        .. code-block:: python

            ie = IECore()
            ie.get_metric(metric_name="SUPPORTED_METRICS", device_name="CPU")
        """
        return self.impl.getMetric(device_name.encode(), metric_name.encode())

    
    def get_config(self, device_name: str, config_name: str):
        """Gets a configuration dedicated to device behavior. The method targets to extract information
        which can be set via set_config method.

        .. note:: When specifying a key value of a config, the "KEY_" prefix is omitted.

        :param device_name: A name of a device to get a config value.
        :param config_name: A config name to request.
        :return: A config value corresponding to a config key.
        
        Usage example:

        .. code-block:: python

            ie = IECore()
            ie.get_config(device_name="CPU", config_name="CPU_BIND_THREAD")
        """
        return self.impl.getConfig(device_name.encode(), config_name.encode())

    ## A list of devices. The devices are returned as \[CPU, GPU.0, GPU.1\].
    # If there are more than one device of a specific type, they all are listed followed by a dot and a number.
    @property
    def available_devices(self):
        """
        A list of devices. The devices are returned as \[CPU, GPU.0, GPU.1\].
        If there are more than one device of a specific type, they all are listed followed by a dot and a number.
        """
        cdef vector[string] c_devices = self.impl.getAvailableDevices()
        return [d.decode() for d in c_devices]

cdef class PreProcessChannel:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This structure stores info about pre-processing of network inputs (scale, mean image, ...)
    """

    property mean_value:
        def __get__(self):
            return deref(self._ptr).meanValue

        def __set__(self, float mean_value):
            deref(self._ptr).meanValue = mean_value
    property std_scale:
        def __get__(self):
            return deref(self._ptr).stdScale

        def __set__(self, float std_scale):
            deref(self._ptr).stdScale = std_scale
    property mean_data:
        def __get__(self):
            blob = Blob()
            blob._ptr = deref(self._ptr).meanData
            return blob

        def __set__(self, Blob mean_data):
            deref(self._ptr).meanData = mean_data._ptr


cdef class PreProcessInfo:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class stores pre-process information for the input
    """

    def __cinit__(self):
        self._ptr = new CPreProcessInfo()
        self._cptr = self._ptr
        self._user_data = True

    def __dealloc__(self):
        if self._user_data:
            del self._ptr

    def __getitem__(self, size_t index):
        cdef CPreProcessChannel.Ptr c_channel = deref(self._cptr)[index]
        channel = PreProcessChannel()
        channel._ptr = c_channel
        return channel


    def get_number_of_channels(self):
        """
        Returns a number of channels to preprocess
        """
        return deref(self._cptr).getNumberOfChannels()


    def init(self, const size_t number_of_channels):
        """
        Initializes with given number of channels
        """
        if not self._ptr:
            raise TypeError("Cannot initialized when created from constant")
        deref(self._ptr).init(number_of_channels)


    def set_mean_image(self, Blob mean_image):
        """
        Sets mean image values if operation is applicable.
        Also sets the mean type to MEAN_IMAGE for all channels
        """
        if not self._ptr:
            raise TypeError("Cannot set mean image when called from constant")
        deref(self._ptr).setMeanImage(mean_image._ptr)


    def set_mean_image_for_channel(self, Blob mean_image, size_t channel):
        """
        Sets mean image values if operation is applicable.
        Also sets the mean type to MEAN_IMAGE for a particular channel
        """
        if not self._ptr:
            raise TypeError("Cannot set mean image for channel when called from constant")
        deref(self._ptr).setMeanImageForChannel(mean_image._ptr, channel)

    @property
    def mean_variant(self):
        """Mean Variant to be applied for input before inference if needed.
        
        Usage example:

        .. code-block:: python
        
            net = ie_core.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            net.input_info['data'].preprocess_info.mean_variant = MeanVariant.MEAN_IMAGE
        """
        return MeanVariant(deref(self._cptr).getMeanVariant())

    @mean_variant.setter
    def mean_variant(self, variant : MeanVariant):
        if not self._ptr:
            raise TypeError("Cannot set mean image when called from constant")
        deref(self._ptr).setVariant(variant.value)

    
    @property
    def resize_algorithm(self):
        """
        Resize Algorithm to be applied for input before inference if needed.
        .. note::
        
            It's need to set your input via the set_blob method.
        
        Usage example:

        .. code-block:: python

            net = ie_core.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            net.input_info['data'].preprocess_info.resize_algorithm = ResizeAlgorithm.RESIZE_BILINEAR
            exec_net = ie_core.load_network(net, 'CPU')
            tensor_desc = ie.TensorDesc("FP32", [1, 3, image.shape[2], image.shape[3]], "NCHW")
            img_blob = ie.Blob(tensor_desc, image)
            request = exec_net.requests[0]
            request.set_blob('data', img_blob)
            request.infer()
        """
        return ResizeAlgorithm(deref(self._cptr).getResizeAlgorithm())

    @resize_algorithm.setter
    def resize_algorithm(self, alg : ResizeAlgorithm):
        if not self._ptr:
            raise TypeError("Cannot set resize algorithm when called from constant")
        deref(self._ptr).setResizeAlgorithm(alg.value)


    @property
    def color_format(self):
        """
        Color format to be used in on-demand color conversions applied to input before inference

        Usage example:

        .. code-block:: python
       
            net = ie_core.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            net.input_info['data'].preprocess_info.color_format = ColorFormat.BGR
        """
        return ColorFormat(deref(self._cptr).getColorFormat())

    @color_format.setter
    def color_format(self, fmt : ColorFormat):
        if not self._ptr:
            raise TypeError("Cannot set color format when called from constant")
        deref(self._ptr).setColorFormat(fmt.value)


cdef class InputInfoPtr:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class contains information about each input of the network
    """

    @property
    def name(self):
        """
        Name of this input
        """
        return deref(self._ptr).name().decode()

    @property
    def precision(self):
        """
        Precision of this input
        """
        return deref(self._ptr).getPrecision().name().decode()

    @precision.setter
    def precision(self, precision : str):
        if precision not in supported_precisions:
            raise ValueError(f"Unsupported precision {precision}! List of supported precisions: {supported_precisions}")
        deref(self._ptr).setPrecision(C.Precision.FromStr(precision.encode()))

    @property
    def layout(self):
        """
        Layout of this input
        """
        return layout_int_to_str_map[deref(self._ptr).getLayout()]

    @layout.setter
    def layout(self, layout : str):
        if layout not in layout_str_to_enum.keys():
            raise ValueError(f"Unsupported layout {layout}! "
                             f"List of supported layouts: {list(layout_str_to_enum.keys())}")
        deref(self._ptr).setLayout(layout_str_to_enum[layout])

   
    @property
    def preprocess_info(self):
        """Gets pre-process info for the input
    
        Usage example:
        
        .. code-block:: python

            net = ie_core.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            net.input_info['data'].preprocess_info.color_format = ColorFormat.BGR
        """
        cdef CPreProcessInfo* c_preprocess_info = &deref(self._ptr).getPreProcess()
        preprocess_info = PreProcessInfo()
        del preprocess_info._ptr
        preprocess_info._user_data = False
        preprocess_info._ptr = c_preprocess_info
        preprocess_info._cptr = c_preprocess_info
        return preprocess_info

    @property
    def tensor_desc(self):
        cdef CTensorDesc c_tensor_desc = deref(self._ptr).getTensorDesc()
        precision = c_tensor_desc.getPrecision().name().decode()
        layout = c_tensor_desc.getLayout()
        dims = c_tensor_desc.getDims()
        tensor_desc = TensorDesc(precision, dims, layout_int_to_str_map[layout])
        tensor_desc.impl = c_tensor_desc
        return tensor_desc

    @property
    def input_data(self):
        """
        Get access to DataPtr object
        """
        cdef C.DataPtr c_data_ptr = deref(self._ptr).getInputData()
        data_ptr = DataPtr()
        data_ptr._ptr_network = self._ptr_network
        data_ptr._ptr = c_data_ptr
        return data_ptr

    @input_data.setter
    def input_data(self, input_ptr : DataPtr):
        deref(self._ptr).setInputData(input_ptr._ptr)


cdef class InputInfoCPtr:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class contains const information about each input of the network.
    Provides same interface as InputInfoPtr object except properties setters
    """

    @property
    def name(self):
        """
        Name of this input
        """
        return deref(self._ptr).name().decode()

    @property
    def precision(self):
        """
        Precision of this input
        """
        return deref(self._ptr).getPrecision().name().decode()

    @property
    def input_data(self):
        """
        Get access to DataPtr object
        """
        cdef C.DataPtr c_data_ptr = deref(self._ptr).getInputData()
        data_ptr = DataPtr()
        data_ptr._ptr = c_data_ptr
        data_ptr._ptr_plugin = self._ptr_plugin
        return data_ptr

    @property
    def tensor_desc(self):
        """
        tensor_desc of this input
        """
        cdef CTensorDesc c_tensor_desc = deref(self._ptr).getTensorDesc()
        precision = c_tensor_desc.getPrecision().name().decode()
        layout = c_tensor_desc.getLayout()
        dims = c_tensor_desc.getDims()
        tensor_desc = TensorDesc(precision, dims, layout_int_to_str_map[layout])
        tensor_desc.impl = c_tensor_desc
        return tensor_desc


cdef class DataPtr:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class is the layer data representation.
    """

    def __init__(self):
        """
        Default constructor
        """
        self._ptr_network = NULL

    @property
    def name(self):
        """
        Name of the data object
        """
        return deref(self._ptr).getName().decode()

    @property
    def precision(self):
        """
        Precision of the data object
        """
        return deref(self._ptr).getPrecision().name().decode()

    @precision.setter
    def precision(self, precision):
        if precision not in supported_precisions:
            raise ValueError(f"Unsupported precision {precision}! List of supported precisions: {supported_precisions}")
        deref(self._ptr).setPrecision(C.Precision.FromStr(precision.encode()))

    @property
    def shape(self):
        """
        Shape (dimensions) of the data object
        """
        return deref(self._ptr).getDims()

    @property
    def layout(self):
        """
        Layout of the data object
        """
        return layout_int_to_str_map[deref(self._ptr).getLayout()]

    @layout.setter
    def layout(self, layout):
        if layout not in layout_str_to_enum.keys():
            raise ValueError(f"Unsupported layout {layout}! "
                             f"List of supported layouts: {list(layout_str_to_enum.keys())}")
        deref(self._ptr).setLayout(layout_str_to_enum[layout])

    @property
    def initialized(self):
        """
        Checks if the current data object is resolved
        """
        return deref(self._ptr).isInitialized()


cdef class CDataPtr:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class is the layer constant data representation. Provides same interface as DataPtr object except properties setters
    """

    @property
    def name(self):
        """
        Name of the data object
        """
        return deref(self._ptr).getName().decode()

    @property
    def precision(self):
        """
        Precision of the data object
        """
        return deref(self._ptr).getPrecision().name().decode()

    @property
    def shape(self):
        """
        Shape (dimensions) of the data object
        """
        return deref(self._ptr).getDims()

    @property
    def layout(self):
        """
        Layout of the data object
        """
        return layout_int_to_str_map[deref(self._ptr).getLayout()]

    @property
    def initialized(self):
        """
        Checks if the current data object is resolved
        """
        return deref(self._ptr).isInitialized()


cdef class ExecutableNetwork:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class represents a network instance loaded to plugin and ready for inference.
    """

    def __init__(self):
        """
        There is no explicit class constructor. To make a valid instance of :class:`ExecutableNetwork`,
        use :func:`IECore.load_network` method of the :class:`IECore` class.
        """
        self._infer_requests = []

    def infer(self, inputs=None):
        """Starts synchronous inference for the first infer request of the executable network and returns output data.
        Wraps :func:`InferRequest.infer` method of the :class:`InferRequest` class
        
        :param inputs:  A dictionary that maps input layer names to :class:`numpy.ndarray` objects of proper shape with
                        input data for the layer
        :return: A dictionary that maps output layer names to :class:`numpy.ndarray` objects with output data of the layer
        
        Usage example:

        .. code-block:: python

            ie_core = IECore()
            net = ie_core.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie_core.load_network(network=net, device_name="CPU", num_requests=2)
            res = exec_net.infer({'data': img})
            res
            {'prob': array([[[[2.83426580e-08]],
                          [[2.40166020e-08]],
                          [[1.29469613e-09]],
                          [[2.95946148e-08]]
                          ......
                         ]])}
        """
        current_request = self.requests[0]
        current_request.infer(inputs)
        res = {}
        for name, value in current_request.output_blobs.items():
            res[name] = deepcopy(value.buffer)
        return res

    def start_async(self, request_id, inputs=None):
        """
        Starts asynchronous inference for specified infer request.
        Wraps :func:`InferRequest.async_infer` method of the :class:`InferRequest` class.
        
        :param request_id: Index of infer request to start inference
        :param inputs: A dictionary that maps input layer names to :class:`numpy.ndarray` objects of proper
                       shape with input data for the layer
        :return: A handler of specified infer request, which is an instance of the :class:`InferRequest` class.

        Usage example:
        
        .. code-block:: python
        
            infer_request_handle = exec_net.start_async(request_id=0, inputs={input_blob: image})
            infer_status = infer_request_handle.wait()
            res = infer_request_handle.output_blobs[out_blob_name]
        """
        if request_id not in list(range(len(self.requests))):
            raise ValueError("Incorrect request_id specified!")
        current_request = self.requests[request_id]
        current_request.async_infer(inputs)
        return current_request


    @property
    def requests(self):
        """
        A tuple of :class:`InferRequest` instances
        """
        cdef int c_infer_requests_size = deref(self.impl).infer_requests.size()
        if len(self._infer_requests) == 0:
            for i in range(c_infer_requests_size):
                infer_request = InferRequest()
                infer_request.impl = &(deref(self.impl).infer_requests[i])
                infer_request._inputs_list = list(self.input_info.keys())
                infer_request._outputs_list = list(self.outputs.keys())
                self._infer_requests.append(infer_request)

        if len(self._infer_requests) != c_infer_requests_size:
            raise Exception("Mismatch of infer requests number!")

        return self._infer_requests

    @property
    def input_info(self):
        """
        A dictionary that maps input layer names to InputInfoCPtr objects
        """
        cdef map[string, C.InputInfo.CPtr] c_inputs = deref(self.impl).getInputsInfo()
        inputs = {}
        cdef InputInfoCPtr input_info_ptr
        for in_ in c_inputs:
            input_info_ptr = InputInfoCPtr()
            input_info_ptr._ptr = in_.second
            input_info_ptr._ptr_plugin = deref(self.impl).getPluginLink()
            inputs[in_.first.decode()] = input_info_ptr
        return inputs

    ## A dictionary that maps output layer names to CDataPtr objects
    @property
    def outputs(self):
        """
        A dictionary that maps output layer names to CDataPtr objects
        """
        cdef map[string, C.CDataPtr] c_outputs = deref(self.impl).getOutputs()
        outputs = {}
        cdef CDataPtr data_ptr
        for in_ in c_outputs:
            data_ptr = CDataPtr()
            data_ptr._ptr = in_.second
            data_ptr._ptr_plugin = deref(self.impl).getPluginLink()
            outputs[in_.first.decode()] = data_ptr
        return outputs


    def get_exec_graph_info(self):
        """Gets executable graph information from a device
        
        :return: An instance of :class:`IENetwork`
        
        Usage example:

        .. code-block:: python
        
            ie_core = IECore()
            net = ie_core.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie_core.load_network(net, device, num_requests=2)
            exec_graph = exec_net.get_exec_graph_info()
        """
        ie_network = IENetwork()
        ie_network.impl = deref(self.impl).GetExecGraphInfo()
        ie_network._ptr_plugin = deref(self.impl).getPluginLink()
        return ie_network


    def get_metric(self, metric_name: str):
        """Gets general runtime metric for an executable network. It can be network name, actual device ID on
        which executable network is running or all other properties which cannot be changed dynamically.
        
        :param metric_name: A metric name to request.
        :return: A metric value corresponding to a metric key.
       
        Usage example:
       
        .. code-block:: python

            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie.load_network(net, "CPU")
            exec_net.get_metric("NETWORK_NAME")
        """
        return deref(self.impl).getMetric(metric_name.encode())

   
    def get_config(self, config_name: str):
        """Gets configuration for current executable network. The method is responsible to extract information
        which affects executable network execution
        
        :param config_name: A configuration parameter name to request.
        :return: A configuration value corresponding to a configuration key.
        
        Usage example:
        
        .. code-block:: python
        
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie.load_network(net, "CPU")
            config = exec_net.get_config("CPU_BIND_THREAD")
        """
        return deref(self.impl).getConfig(config_name.encode())

    ## Sets configuration for current executable network.
    #
    #  @param config: a dictionary of configuration parameters as keys and their values
    #  @return None
    #
    #  Usage example:\n
    #  ```python
    #  ie = IECore()
    #  net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
    #  exec_net = ie.load_network(net, "GNA")
    #  config = exec_net.set_config({"DEVICE_MODE" : "GNA_SW_EXACT"})
    #  ```
    def set_config(self, config: dict):
        cdef map[string, string] c_config = dict_to_c_map(config)
        deref(self.impl).setConfig(c_config)

    ## Exports the current executable network.
    #  @param model_file Full path to the target exported file location
    #  @return None
    #
    #  ```python
    #  ie = IECore()
    #  net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
    #  exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
    #  exec_net.export(path_to_file_to_save)
    #  ```
    def export(self, model_file: str):
        """Exports the current executable network.
        
        :param model_file: Full path to the target exported file location
        :return: None

        .. code-block:: python
    
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
            exec_net.export(path_to_file_to_save)
        """
        deref(self.impl).exportNetwork(model_file.encode())

    cpdef wait(self, num_requests=None, timeout=None):
        """Waits when the result from any request becomes available. Blocks until specified timeout elapses or the result.
        
        :param num_requests: Number of idle requests for which wait.
                             If not specified, `num_requests` value is set to number of requests by default.
        :param timeout: Time to wait in milliseconds or special (0, -1) cases described above.
                        If not specified, `timeout` value is set to -1 by default.
        :return: Request status code: `OK` or `RESULT_NOT_READY`
        """
        cdef int status_code
        cdef int64_t c_timeout
        cdef int c_num_requests
        if num_requests is None:
            num_requests = len(self.requests)
        c_num_requests = <int> num_requests
        if timeout is None:
            timeout = WaitMode.RESULT_READY
        c_timeout = <int64_t> timeout
        with nogil:
            status_code = deref(self.impl).wait(c_num_requests, c_timeout)
        return status_code

    
    cpdef get_idle_request_id(self):
        """
        Get idle request ID
        
        :return: Request index
        """
        return deref(self.impl).getIdleRequestId()

ctypedef extern void (*cb_type)(void*, int) with gil


cdef class InferRequest:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    This class provides an interface to infer requests of :class:`ExecutableNetwork` and serves
    to handle infer requests execution and to set and get output data.
    """

    def __init__(self):
        """
        There is no explicit class constructor. To make a valid :class:`InferRequest` instance, use :func:`IECore.load_network`
        method of the :class:`IECore` class with specified number of requests to get :class:`ExecutableNetwork` instance
        which stores infer requests.
        """
        self._user_blobs = {}
        self._inputs_list = []
        self._outputs_list = []
        self._py_callback = lambda *args, **kwargs: None
        self._py_data = None

    cdef void user_callback(self, int status) with gil:
        if self._py_callback:
            self._py_callback(status, self._py_data)

    def set_completion_callback(self, py_callback, py_data = None):
        """Description: Sets a callback function that is called on success or failure of an asynchronous request
        
        :param py_callback: Any defined or lambda function
        :param py_data: Data that is passed to the callback function
        :return: None
        
        Usage example:

        .. code-block:: python
        
            callback = lambda status, py_data: print(f"Request with id {py_data} finished with status {status}")
            ie = IECore()
            net = ie.read_network(model="./model.xml", weights="./model.bin")
            exec_net = ie.load_network(net, "CPU", num_requests=4)
            for id, req in enumerate(exec_net.requests):
                req.set_completion_callback(py_callback=callback, py_data=id)
                
            for req in exec_net.requests:
                req.async_infer({"data": img})
        """
        self._py_callback = py_callback
        self._py_data = py_data
        deref(self.impl).setCyCallback(<cb_type> self.user_callback, <void *> self)

    cpdef BlobBuffer _get_blob_buffer(self, const string & blob_name):
        cdef BlobBuffer buffer = BlobBuffer()
        cdef CBlob.Ptr blob_ptr
        blob_ptr = deref(self.impl).getBlobPtr(blob_name)
        buffer.reset(blob_ptr)
        return buffer


    @property
    def input_blobs(self):
        """
        Dictionary that maps input layer names to corresponding Blobs
        """
        input_blobs = {}
        for input in self._inputs_list:
            # TODO: will not work for setting data via .inputs['data'][:]
            if input in self._user_blobs:
                input_blobs[input] = self._user_blobs[input]
            else:
                blob = Blob()
                blob._ptr = deref(self.impl).getBlobPtr(input.encode())
                input_blobs[input] = blob
        return input_blobs

    @property
    def output_blobs(self):
        """
        Dictionary that maps output layer names to corresponding Blobs
        """
        output_blobs = {}
        for output in self._outputs_list:
            blob = Blob()
            blob._ptr = deref(self.impl).getBlobPtr(output.encode())
            output_blobs[output] = deepcopy(blob)
        return output_blobs

    @property
    def preprocess_info(self):
        """
        Dictionary that maps input layer names to corresponding preprocessing information
        """
        preprocess_info = {}
        for input_blob in self.input_blobs.keys():
            preprocess = PreProcessInfo()
            del preprocess._ptr
            preprocess._user_data = False
            preprocess._ptr = NULL
            preprocess._cptr = &deref(self.impl).getPreProcess(input_blob.encode())
            preprocess_info[input_blob] = preprocess
        return preprocess_info

    def query_state(self):
        """Gets state control interface for given infer request
        State control essential for recurrent networks
        :return: A vector of Memory State objects
        """
        cdef vector[C.CVariableState] c_mem_state_vec = deref(self.impl).queryState()
        mem_state_vec = []
        for ms in c_mem_state_vec:
            state = VariableState()
            state.impl = ms
            mem_state_vec.append(state)
        return mem_state_vec

    def set_blob(self, blob_name : str, blob : Blob):
        """Sets user defined Blob for the infer request
        
        :param blob_name: A name of input blob
        :param blob: Blob object to set for the infer request
        :param preprocess_info: PreProcessInfo object to set for the infer request.
        :return: None
        
        Usage example:

        .. code-block:: python
    
            ie = IECore()
            net = IENetwork("./model.xml", "./model.bin")
            exec_net = ie.load_network(net, "CPU", num_requests=2)
            td = TensorDesc("FP32", (1, 3, 224, 224), "NCHW")
            blob_data = np.ones(shape=(1, 3, 224, 224), dtype=np.float32)
            blob = Blob(td, blob_data)
            exec_net.requests[0].set_blob(blob_name="input_blob_name", blob=blob),
        """
        deref(self.impl).setBlob(blob_name.encode(), blob._ptr)
        self._user_blobs[blob_name] = blob

    cpdef infer(self, inputs=None):
        """Starts synchronous inference of the infer request and fill outputs array
        
        :param inputs: A dictionary that maps input layer names to :class:`numpy.ndarray` objects of proper shape with
                       input data for the layer
        :return: None

        Usage example:

        .. code-block:: python
    
            exec_net = ie_core.load_network(network=net, device_name="CPU", num_requests=2)
            exec_net.requests[0].infer({input_blob: image})
            res = exec_net.requests[0].output_blobs['prob']
            np.flip(np.sort(np.squeeze(res)),0)

            # array([4.85416055e-01, 1.70385033e-01, 1.21873841e-01, 1.18894853e-01,
            #         5.45198545e-02, 2.44456064e-02, 5.41366823e-03, 3.42589128e-03,
            #         2.26027006e-03, 2.12283316e-03 ...])
        """
        if inputs is not None:
            self._fill_inputs(inputs)
        deref(self.impl).infer()

    cpdef async_infer(self, inputs=None):
        """Starts asynchronous inference of the infer request and fill outputs array
        
        :param inputs: A dictionary that maps input layer names to :class:`numpy.ndarray` objects
                       of proper shape with input data for the layer
        :return: None
        
        Usage example:
    
        .. code-block:: python
        
            exec_net = ie_core.load_network(network=net, device_name="CPU", num_requests=2)
            exec_net.requests[0].async_infer({input_blob: image})
            request_status = exec_net.requests[0].wait()
            res = exec_net.requests[0].output_blobs['prob']
        """
        if inputs is not None:
            self._fill_inputs(inputs)
        deref(self.impl).infer_async()

    cpdef wait(self, timeout=None):
        """Waits for the result to become available. Blocks until specified timeout elapses or the result
        becomes available, whichever comes first.

        :param timeout: Time to wait in milliseconds or special (0, -1) cases described above.
                        If not specified, `timeout` value is set to -1 by default.
        :return: Request status code.

        .. note::
        
            There are special values of the timeout parameter:
            
            * 0 - Immediately returns the inference status. It does not block or interrupt execution.
              To find statuses meaning, please refer to :ref:`enum_InferenceEngine_StatusCode` in Inference Engine C++ documentation
            * -1 - Waits until inference result becomes available (default value)
    
        Usage example: See :func:`InferRequest.async_infer` method of the the :class:`InferRequest` class.
        """
        cdef int status
        cdef int64_t c_timeout
        if timeout is None:
            timeout = WaitMode.RESULT_READY
        c_timeout = <int64_t> timeout
        with nogil:
            status = deref(self.impl).wait(c_timeout)
        return status


    cpdef get_perf_counts(self):
        """Queries performance measures per layer to get feedback of what is the most time consuming layer.
        
        .. note:: Performance counters data and format depends on the plugin

        :return: Dictionary containing per-layer execution information.
    
        Usage example:
        
        .. code-block:: python

            exec_net = ie_core.load_network(network=net, device_name="CPU", num_requests=2)
            exec_net.requests[0].infer({input_blob: image})
            exec_net.requests[0].get_perf_counts()
            #  {'Conv2D': {'exec_type': 'jit_avx2_1x1',
            #              'real_time': 154,
            #              'cpu_time': 154,
            #              'status': 'EXECUTED',
            #              'layer_type': 'Convolution'},
            #   'Relu6':  {'exec_type': 'undef',
            #              'real_time': 0,
            #              'cpu_time': 0,
            #              'status': 'NOT_RUN',
            #              'layer_type': 'Clamp'}
            #   ...
            #  }
        """
        cdef map[string, C.ProfileInfo] c_profile = deref(self.impl).getPerformanceCounts()
        profile = {}
        for line in c_profile:
            info = line.second
            # TODO: add execution index. Check if unsigned int is properly converted to int in python.
            profile[line.first.decode()] = {"status": info.status.decode(), "exec_type": info.exec_type.decode(),
                                            "layer_type": info.layer_type.decode(), "real_time": info.real_time,
                                            "cpu_time": info.cpu_time, "execution_index": info.execution_index}
        return profile

    ## Current infer request inference time in milliseconds
    @property
    def latency(self):
        """
        Current infer request inference time in milliseconds
        """
        return self.impl.exec_time


    def _fill_inputs(self, inputs):
        for k, v in inputs.items():
            assert k in self._inputs_list, f"No input with name {k} found in network"
            if self.input_blobs[k].tensor_desc.precision == "FP16":
                self.input_blobs[k].buffer[:] = v.view(dtype=np.int16)
            else:
                self.input_blobs[k].buffer[:] = v


cdef class IENetwork:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
    """
    ## Class constructor
    #
    #  @param model: A PyCapsule containing smart pointer to nGraph function.
    #
    #  @return Instance of IENetwork class
    #
    #  Usage example:\n
    #   Initializing `IENetwork` object from IR files:
    #   ```python
    #   func = Function([relu], [param], 'test')
    #   caps = Function.to_capsule(func)
    #   net = IENetwork(caps)
    #   ```
    def __cinit__(self, model = None):
        # Try to create Inference Engine network from capsule
        if model is not None:
            self.impl = C.IENetwork(model)
        else:
            with nogil:
                self.impl = C.IENetwork()

    @property
    def name(self):
        """
        Name of the loaded network
        """
        name = bytes(self.impl.name)
        return name.decode()

    @property
    def input_info(self):
        """
        A dictionary that maps input layer names to InputInfoPtr objects.
        """
        cdef map[string, C.InputInfo.Ptr] c_inputs = self.impl.getInputsInfo()
        inputs = {}
        cdef InputInfoPtr input_info_ptr
        for input in c_inputs:
            input_info_ptr = InputInfoPtr()
            input_info_ptr._ptr = input.second
            input_info_ptr._ptr_network = &self.impl
            inputs[input.first.decode()] = input_info_ptr
        return inputs

    ## A dictionary that maps output layer names to DataPtr objects
    @property
    def outputs(self):
        """
        A dictionary that maps output layer names to DataPtr objects
        """
        cdef map[string, C.DataPtr] c_outputs = self.impl.getOutputs()
        outputs = {}
        cdef DataPtr data_ptr
        for output in c_outputs:
            data_ptr = DataPtr()
            data_ptr._ptr_network = &self.impl
            data_ptr._ptr = output.second
            outputs[output.first.decode()] = data_ptr
        return outputs

    
    @property
    def batch_size(self):
        """Batch size of the network. Provides getter and setter interfaces to get and modify the
        network batch size. For example:
       
        .. code-block:: python
        
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            print(net.batch_size)
            net.batch_size = 4
            print(net.batch_size)
            print(net.input_info['data'].input_data.shape)
        """
        return self.impl.getBatch()

    @batch_size.setter
    def batch_size(self, batch: int):
        if batch <= 0:
            raise AttributeError(f"Invalid batch size {batch}! Batch size should be positive integer value")
        self.impl.setBatch(batch)

    def add_outputs(self, outputs):
        """Marks any intermediate layer as output layer to retrieve the inference results from the specified layers.
        
        :param outputs: List of layers to be set as model outputs. The list can contain strings with layer names to be set
                        as outputs or tuples with layer name as first element and output port id as second element.
                        In case of setting one layer as output, string or tuple with one layer can be provided.
        
        :return: None

        Usage example:
    
        .. code-block:: python
    
            ie = IECore()
            net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
            net.add_outputs(["conv5_1', conv2_1', (split_2, 1)])]
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        for i, line in enumerate(outputs):
            if isinstance(line, str):
                self.impl.addOutput(line.encode(), 0)
            elif isinstance(line, tuple) and len(line) == 2:
                self.impl.addOutput(line[0].encode(), line[1])
            else:
                raise TypeError(f"Incorrect type {type(line)} for layer to add at index {i}. "
                                "Expected string with layer name or tuple with two elements: layer name as "
                                "first element and port id as second")

    def serialize(self, path_to_xml, path_to_bin: str = ""):
        """Serializes the network and stores it in files.
       
        :param path_to_xml: Path to a file, where a serialized model will be stored
        :param path_to_bin: Path to a file, where serialized weights will be stored
        :return: None
    
        Usage example:
    
        .. code-block:: python

            ie = IECore()
            net = ie.read_network(model=path_to_xml, weights=path_to_bin)
            net.serialize(path_to_xml, path_to_bin)
        """
        self.impl.serialize(path_to_xml.encode(), path_to_bin.encode())

    def reshape(self, input_shapes: dict):
        """Reshapes the network to change spatial dimensions, batch size, or any dimension.
        
        :param input_shapes: A dictionary that maps input layer names to tuples with the target shape
        :return: None

        .. note::
        
           Before using this method, make sure that the target shape is applicable for the network.
           Changing the network shape to an arbitrary value may lead to unpredictable behaviour.

        Usage example:
       
        .. code-block:: python
    
           ie = IECore()
           net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
           input_layer = next(iter(net.input_info))
           n, c, h, w = net.input_info[input_layer].input_data.shape
           net.reshape({input_layer: (n, c, h*2, w*2)})
        """
        cdef map[string, vector[size_t]] c_input_shapes
        cdef vector[size_t] c_shape
        net_inputs = self.input_info
        for input, shape in input_shapes.items():
            c_shape = []
            if input not in net_inputs:
                raise AttributeError(f"Specified '{input}' layer not in network inputs '{net_inputs}'! ")
            for v in shape:
                try:
                    c_shape.push_back(v)
                except OverflowError:
                    raise ValueError(f"Detected dynamic dimension in the shape {shape} of the `{input}` input. Dynamic shapes are supported since OpenVINO Runtime API 2022.1.")

            c_input_shapes[input.encode()] = c_shape
        self.impl.reshape(c_input_shapes)

    def _get_function_capsule(self):
        return self.impl.getFunction()

    def get_ov_name_for_tensor(self, orig_name: str):
        name = bytes(orig_name, 'utf-8')
        return self.impl.getOVNameForTensor(name).decode('utf-8')

cdef class BlobBuffer:
    """
    OpenVINO Inference Engine Python API is deprecated and will be removed in the 2024.0 release. For instructions on
    transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html

    Copy-less accessor for Inference Engine Blob
    """

    cdef reset(self, CBlob.Ptr & ptr, vector[size_t] representation_shape = []):
        self.ptr = ptr
        cdef CTensorDesc desc = deref(ptr).getTensorDesc()
        cdef SizeVector shape
        if len(representation_shape) == 0:
            shape = desc.getDims()
            if layout_int_to_str_map[desc.getLayout()] == 'SCALAR':
                shape = [1]
        else:
            shape = representation_shape
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

    cdef char*_get_blob_format(self, const CTensorDesc & desc):
        cdef Precision precision = desc.getPrecision()
        name = bytes(precision.name()).decode()
        # todo: half floats
        precision_to_format = {
            'FP32': 'f',  # float
            'FP64': 'd',  # double
            'FP16': 'h',  # signed short
            'U8': 'B',  # unsigned char
            'U16': 'H',  # unsigned short
            'I8': 'b',  # signed char
            'I16': 'h',  # signed short
            'I32': 'i',  # signed int
            'U32': 'I',  # unsigned int
            'I64': 'q',  # signed long int
            'U64': 'Q',  # unsigned long int
            'BOOL': 'B',  # unsigned char
            'BF16': 'h',  # signed short
            'BIN': 'b',  # signed char
        }
        if name not in precision_to_format:
            raise ValueError(f"Unknown Blob precision: {name}")

        return precision_to_format[name].encode()

    def to_numpy(self, is_const= False):
        precision = deref(self.ptr).getTensorDesc().getPrecision()
        name = bytes(precision.name()).decode()
        arr = np.asarray(self)
        if is_const:
            arr.flags.writeable = False
        if name == "FP16":
            return arr.view(dtype=np.float16)
        else:
            return arr
