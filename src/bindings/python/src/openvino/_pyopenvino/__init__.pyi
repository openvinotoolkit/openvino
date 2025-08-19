# type: ignore
from . import _offline_transformations
from . import experimental
from . import frontend
from . import layout_helpers
from . import op
from . import passes
from . import preprocess
from . import properties
from . import util
from __future__ import annotations
import collections.abc
import datetime
import numpy
import typing
"""
Package openvino._pyopenvino which wraps openvino C++ APIs
"""
__all__ = ['AsyncInferQueue', 'AttributeVisitor', 'AxisSet', 'AxisVector', 'CompiledModel', 'ConstOutput', 'ConversionExtension', 'ConversionExtensionBase', 'Coordinate', 'CoordinateDiff', 'Core', 'DecoderTransformationExtension', 'DescriptorTensor', 'Dimension', 'DiscreteTypeInfo', 'Extension', 'FrontEnd', 'FrontEndManager', 'GeneralFailure', 'InferRequest', 'InitializationFailure', 'Input', 'InputModel', 'Iterator', 'Layout', 'Model', 'Node', 'NodeContext', 'NodeFactory', 'NotImplementedFailure', 'OVAny', 'Op', 'OpConversionFailure', 'OpExtension', 'OpValidationFailure', 'Output', 'PartialShape', 'Place', 'ProfilingInfo', 'ProgressReporterExtension', 'RTMap', 'RemoteContext', 'RemoteTensor', 'Shape', 'Strides', 'Symbol', 'TelemetryExtension', 'Tensor', 'Type', 'VAContext', 'VASurfaceTensor', 'VariableState', 'Version', 'experimental', 'frontend', 'get_batch', 'get_version', 'layout_helpers', 'op', 'passes', 'preprocess', 'properties', 'save_model', 'serialize', 'set_batch', 'shutdown', 'util']
class AsyncInferQueue:
    """
    openvino.AsyncInferQueue represents a helper that creates a pool of asynchronousInferRequests and provides synchronization functions to control flow of a simple pipeline.
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> InferRequest:
        """
                :param i: InferRequest id
                :type i: int
                :return: InferRequests from the pool with given id.
                :rtype: openvino.InferRequest
        """
    def __init__(self, model: CompiledModel, jobs: typing.SupportsInt = 0) -> None:
        """
                        Creates AsyncInferQueue.
        
                        :param model: Model to be used to create InferRequests in a pool.
                        :type model: openvino.CompiledModel
                        :param jobs: Number of InferRequests objects in a pool. If 0, jobs number
                        will be set automatically to the optimal number. Default: 0
                        :type jobs: int
                        :rtype: openvino.AsyncInferQueue
        """
    def __iter__(self) -> collections.abc.Iterator[InferRequest]:
        ...
    def __len__(self) -> int:
        """
                Number of InferRequests in the pool.
        
                :rtype: int
        """
    def __repr__(self) -> str:
        ...
    def get_idle_request_id(self) -> int:
        """
                    Returns next free id of InferRequest from queue's pool.
                    Function waits for any request to complete and then returns this request's id.
        
                    GIL is released while running this function.
        
                    :rtype: int
        """
    def is_ready(self) -> bool:
        """
                    One of 'flow control' functions.
                    Returns True if any free request in the pool, otherwise False.
        
                    GIL is released while running this function.
        
                    :return: If there is at least one free InferRequest in a pool, returns True.
                    :rtype: bool
        """
    def set_callback(self, arg0: collections.abc.Callable) -> None:
        """
                    Sets unified callback on all InferRequests from queue's pool.
                    Signature of such function should have two arguments, where
                    first one is InferRequest object and second one is userdata
                    connected to InferRequest from the AsyncInferQueue's pool.
        
                    .. code-block:: python
        
                        def f(request, userdata):
                            result = request.output_tensors[0]
                            print(result + userdata)
        
                        async_infer_queue.set_callback(f)
        
                    :param callback: Any Python defined function that matches callback's requirements.
                    :type callback: function
        """
    @typing.overload
    def start_async(self, inputs: Tensor, userdata: typing.Any) -> None:
        """
                    Run asynchronous inference using the next available InferRequest.
        
                    This function releases the GIL, so another Python thread can
                    work while this function runs in the background.
        
                    :param inputs: Data to set on single input tensor of next available InferRequest from
                    AsyncInferQueue's pool.
                    :type inputs: openvino.Tensor
                    :param userdata: Any data that will be passed to a callback
                    :type userdata: Any
                    :rtype: None
        
                    GIL is released while waiting for the next available InferRequest.
        """
    @typing.overload
    def start_async(self, inputs: dict, userdata: typing.Any) -> None:
        """
                    Run asynchronous inference using the next available InferRequest.
        
                    This function releases the GIL, so another Python thread can
                    work while this function runs in the background.
        
                    :param inputs: Data to set on input tensors of next available InferRequest from
                    AsyncInferQueue's pool.
                    :type inputs: dict[Union[int, str, openvino.ConstOutput] : openvino.Tensor]
                    :param userdata: Any data that will be passed to a callback
                    :rtype: None
        
                    GIL is released while waiting for the next available InferRequest.
        """
    def wait_all(self) -> None:
        """
                    One of 'flow control' functions. Blocking call.
                    Waits for all InferRequests in a pool to finish scheduled work.
        
                    GIL is released while running this function.
        """
    @property
    def userdata(self) -> list[typing.Any]:
        """
                :return: list of all passed userdata. list is filled with `None` if the data wasn't passed yet.
                :rtype: list[Any]
        """
class AttributeVisitor:
    def on_attributes(self, arg0: dict) -> None:
        ...
class AxisSet:
    """
    openvino.AxisSet wraps ov::AxisSet
    """
    @typing.overload
    def __init__(self, axes: collections.abc.Set[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, axes: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, axes: AxisSet) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
class AxisVector:
    """
    openvino.AxisVector wraps ov::AxisVector
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __init__(self, axes: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, axes: AxisVector) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
class CompiledModel:
    """
    openvino.CompiledModel represents Model that is compiled for a specific device by applying multiple optimization transformations, then mapping to compute kernels.
    """
    def __init__(self, other: CompiledModel) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def create_infer_request(self) -> InferRequestWrapper:
        """
                    Creates an inference request object used to infer the compiled model.
                    The created request has allocated input and output tensors.
        
                    :return: New InferRequest object.
                    :rtype: openvino.InferRequest
        """
    @typing.overload
    def export_model(self) -> typing.Any:
        """
                    Exports the compiled model to bytes/output stream.
        
                    GIL is released while running this function.
        
                    :return: Bytes object that contains this compiled model.
                    :rtype: bytes
        
                    .. code-block:: python
        
                        user_stream = compiled.export_model()
        
                        with open('./my_model', 'wb') as f:
                            f.write(user_stream)
        
                        # ...
        
                        new_compiled = core.import_model(user_stream, "CPU")
        """
    @typing.overload
    def export_model(self, model_stream: typing.Any) -> None:
        """
                    Exports the compiled model to bytes/output stream.
        
                    Advanced version of `export_model`. It utilizes, streams from the standard
                    Python library `io`.
        
                    Function performs flushing of the stream, writes to it, and then rewinds
                    the stream to the beginning (using seek(0)).
        
                    GIL is released while running this function.
        
                    :param model_stream: A stream object to which the model will be serialized.
                    :type model_stream: io.BytesIO
                    :rtype: None
        
                    .. code-block:: python
        
                        user_stream = io.BytesIO()
                        compiled.export_model(user_stream)
        
                        with open('./my_model', 'wb') as f:
                            f.write(user_stream.getvalue()) # or read() if seek(0) was applied before
        
                        # ...
        
                        new_compiled = core.import_model(user_stream, "CPU")
        """
    def get_property(self, property: str) -> typing.Any:
        """
                    Gets properties for current compiled model.
        
                    :param name: Property name.
                    :type name: str
                    :rtype: Any
        """
    def get_runtime_model(self) -> Model:
        """
                        Gets runtime model information from a device.
        
                        This object (returned model) represents the internal device-specific model
                        which is optimized for the particular accelerator. It contains device-specific nodes,
                        runtime information, and can be used only to understand how the source model
                        is optimized and which kernels, element types, and layouts are selected.
        
                        :return: Model, containing Executable Graph information.
                        :rtype: openvino.Model
        """
    @typing.overload
    def input(self) -> ConstOutput:
        """
                        Gets a single input of a compiled model.
                        If a model has more than one input, this method throws an exception.
        
                        :return: A compiled model input.
                        :rtype: openvino.ConstOutput
        """
    @typing.overload
    def input(self, index: typing.SupportsInt) -> ConstOutput:
        """
                        Gets input of a compiled model identified by an index.
                        If the input with given index is not found, this method throws an exception.
        
                        :param index: An input index.
                        :type index: int
                        :return: A compiled model input.
                        :rtype: openvino.ConstOutput
        """
    @typing.overload
    def input(self, tensor_name: str) -> ConstOutput:
        """
                        Gets input of a compiled model identified by a tensor_name.
                        If the input with given tensor name is not found, this method throws an exception.
        
                        :param tensor_name: An input tensor name.
                        :type tensor_name: str
                        :return: A compiled model input.
                        :rtype: openvino.ConstOutput
        """
    @typing.overload
    def output(self) -> ConstOutput:
        """
                        Gets a single output of a compiled model.
                        If the model has more than one output, this method throws an exception.
        
                        :return: A compiled model output.
                        :rtype: openvino.ConstOutput
        """
    @typing.overload
    def output(self, index: typing.SupportsInt) -> ConstOutput:
        """
                        Gets output of a compiled model identified by an index.
                        If the output with given index is not found, this method throws an exception.
        
                        :param index: An output index.
                        :type index: int
                        :return: A compiled model output.
                        :rtype: openvino.ConstOutput
        """
    @typing.overload
    def output(self, tensor_name: str) -> ConstOutput:
        """
                        Gets output of a compiled model identified by a tensor_name.
                        If the output with given tensor name is not found, this method throws an exception.
        
                        :param tensor_name: An output tensor name.
                        :type tensor_name: str
                        :return: A compiled model output.
                        :rtype: openvino.ConstOutput
        """
    def release_memory(self) -> None:
        """
                        Release intermediate memory.
        
                        This method forces the Compiled model to release memory allocated for intermediate structures,
                        e.g. caches, tensors, temporal buffers etc., when possible
        """
    @typing.overload
    def set_property(self, properties: collections.abc.Mapping[str, typing.Any]) -> None:
        """
                    Sets properties for current compiled model.
        
                    :param properties: dict of pairs: (property name, property value)
                    :type properties: dict
                    :rtype: None
        """
    @typing.overload
    def set_property(self, property: tuple[str, typing.Any]) -> None:
        """
                    Sets properties for current compiled model.
        
                    :param property: tuple of (property name, matching property value).
                    :type property: tuple
        """
    @property
    def inputs(self) -> list[ConstOutput]:
        """
                                        Gets all inputs of a compiled model.
        
                                        :return: Inputs of a compiled model.
                                        :rtype: list[openvino.ConstOutput]
        """
    @property
    def outputs(self) -> list[ConstOutput]:
        """
                                        Gets all outputs of a compiled model.
        
                                        :return: Outputs of a compiled model.
                                        :rtype: list[openvino.ConstOutput]
        """
class ConstOutput:
    """
    openvino.ConstOutput represents port/node output.
    """
    def __copy__(self) -> ConstOutput:
        ...
    def __deepcopy__(self, arg0: dict) -> None:
        ...
    def __eq__(self, arg0: ConstOutput) -> bool:
        ...
    def __ge__(self, arg0: ConstOutput) -> bool:
        ...
    def __gt__(self, arg0: ConstOutput) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __le__(self, arg0: ConstOutput) -> bool:
        ...
    def __lt__(self, arg0: ConstOutput) -> bool:
        ...
    def __ne__(self, arg0: ConstOutput) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def _from_node(self: Node) -> Output:
        ...
    def get_any_name(self) -> str:
        """
                        One of the tensor names associated with this output.
                        Note: first name in lexicographical order.
        
                        :return: Tensor name as string.
                        :rtype: str
        """
    def get_element_type(self) -> Type:
        """
                        The element type of the output referred to by this output handle.
        
                        :return: Type of the output.
                        :rtype: openvino.Type
        """
    def get_index(self) -> int:
        """
                        The index of the output referred to by this output handle.
        
                        :return: Index value as integer.
                        :rtype: int
        """
    def get_names(self) -> set[str]:
        """
                        The tensor names associated with this output.
        
                        :return: set of tensor names.
                        :rtype: set[str]
        """
    def get_node(self) -> Node:
        """
                        Get node referenced by this output handle.
        
                        :return: Node object referenced by this output handle.
                        :rtype: openvino.Node
        """
    def get_partial_shape(self) -> PartialShape:
        """
                        The partial shape of the output referred to by this output handle.
        
                        :return: Copy of PartialShape of the output.
                        :rtype: openvino.PartialShape
        """
    def get_rt_info(self) -> RTMap:
        """
                        Returns RTMap which is a dictionary of user defined runtime info.
        
                        :return: A dictionary of user defined data.
                        :rtype: openvino.RTMap
        """
    def get_shape(self) -> Shape:
        """
                        The shape of the output referred to by this output handle.
        
                        :return: Copy of Shape of the output.
                        :rtype: openvino.Shape
        """
    def get_target_inputs(self) -> set[Input]:
        """
                        A set containing handles for all inputs, targeted by the output,
                        referenced by this output handle.
        
                        :return: set of Inputs.
                        :rtype: set[openvino.Input]
        """
    def get_tensor(self) -> DescriptorTensor:
        """
                        A reference to the tensor descriptor for this output.
        
                        :return: Tensor of the output.
                        :rtype: openvino._pyopenvino.DescriptorTensor
        """
    @property
    def any_name(self) -> str:
        ...
    @property
    def element_type(self) -> Type:
        ...
    @property
    def index(self) -> int:
        ...
    @property
    def names(self) -> set[str]:
        ...
    @property
    def node(self) -> Node:
        ...
    @property
    def partial_shape(self) -> PartialShape:
        ...
    @property
    def rt_info(self) -> RTMap:
        ...
    @property
    def shape(self) -> Shape:
        ...
    @property
    def target_inputs(self) -> set[Input]:
        ...
    @property
    def tensor(self) -> DescriptorTensor:
        ...
class ConversionExtension(_ConversionExtension):
    @typing.overload
    def __init__(self, arg0: str, arg1: collections.abc.Callable[[NodeContext], list[Output]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: collections.abc.Callable[[NodeContext], dict[str, list[Output]]]) -> None:
        ...
class ConversionExtensionBase(Extension):
    pass
class Coordinate:
    """
    openvino.Coordinate wraps ov::Coordinate
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __init__(self, arg0: Shape) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Coordinate) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
class CoordinateDiff:
    """
    openvino.CoordinateDiff wraps ov::CoordinateDiff
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: CoordinateDiff) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
class Core:
    """
    openvino.Core class represents OpenVINO runtime Core entity. User applications can create several Core class instances, but in this case, the underlying plugins are created multiple times and not shared between several Core instances. The recommended way is to have a single Core instance per application.
    """
    def __init__(self, xml_config_file: str = '') -> None:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def add_extension(self, library_path: str) -> None:
        """
                        Registers an extension to a Core object.
        
                        :param library_path: Path to library with ov::Extension
                        :type library_path: str
        """
    @typing.overload
    def add_extension(self, extension: Extension) -> None:
        """
                        Registers an extension to a Core object.
        
                        :param extension: Extension object.
                        :type extension: openvino.Extension
        """
    @typing.overload
    def add_extension(self, extensions: collections.abc.Sequence[Extension]) -> None:
        """
                    Registers extensions to a Core object.
        
                    :param extensions: list of Extension objects.
                    :type extensions: list[openvino.Extension]
        """
    @typing.overload
    def add_extension(self, custom_op: typing.Any) -> None:
        """
                    Registers custom Op to a Core object.
        
                    :param custom_op: type of custom Op
                    :type custom_op: type[openvino.Op]
        """
    @typing.overload
    def compile_model(self, model: Model, device_name: str, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Creates a compiled model from a source model object.
                    Users can create as many compiled models as they need, and use them simultaneously
                    (up to the limitation of the hardware resources).
        
                    GIL is released while running this function.
        
                    :param model: Model acquired from read_model function.
                    :type model: openvino.Model
                    :param device_name: Name of the device which will load the model.
                    :type device_name: str
                    :param properties: Optional dict of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict[str, typing.Any]
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        """
    @typing.overload
    def compile_model(self, model: Model, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Creates and loads a compiled model from a source model to the default OpenVINO device
                    selected by AUTO plugin. Users can create as many compiled models as they need, and use
                    them simultaneously (up to the limitation of the hardware resources).
        
                    GIL is released while running this function.
        
                    :param model: Model acquired from read_model function.
                    :type model: openvino.Model
                    :param properties: Optional dict of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict[str, typing.Any]
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        """
    @typing.overload
    def compile_model(self, model_path: typing.Any, device_name: str, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Reads model and creates a compiled model from IR / ONNX / PDPD / TF and TFLite file.
                    This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
                    especially for cases when caching is enabled and cached model is available.
        
                    GIL is released while running this function.
        
                    :param model_path: A path to a model in IR / ONNX / PDPD / TF and TFLite format.
                    :type model_path: typing.Union[str, pathlib.Path]
                    :param device_name: Name of the device to load the model to.
                    :type device_name: str
                    :param properties: Optional dict of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict[str, typing.Any]
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        """
    @typing.overload
    def compile_model(self, model_buffer: typing.Any, weight_buffer: typing.Any, device_name: str, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Create a compiled model from IR model buffer and weight buffer in memory.
                    This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
                    especially for cases when caching is enabled and cached model is available.
        
                    GIL is released while runing this function.
        
                    :param model_buffer: A string buffer of IR xml in memory
                    :type model_buffer: str
                    :param weight_buffer: A byte buffer of IR weights in memory
                    :type weight_buffer: bytes
                    :param device_name: Name of the device to load the model to.
                    :type device_name: str
                    :param properties: Optional dict of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        """
    @typing.overload
    def compile_model(self, model_path: typing.Any, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Reads model and creates a compiled model from IR / ONNX / PDPD / TF and TFLite file with device selected by AUTO plugin.
                    This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
                    especially for cases when caching is enabled and cached model is available.
        
                    GIL is released while running this function.
        
                    :param model_path: A path to a model in IR / ONNX / PDPD / TF and TFLite format.
                    :type model_path: typing.Union[str, pathlib.Path]
                    :param properties: Optional dict of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict[str, typing.Any]
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        """
    @typing.overload
    def compile_model(self, model: Model, context: RemoteContext, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Creates a compiled model from a source model within a specified remote context.
        
                    GIL is released while running this function.
        
                    :param model: Model acquired from read_model function.
                    :type model: openvino.Model
                    :param context: RemoteContext instance.
                    :type context: openvino.RemoteContext
                    :param properties: dict of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict[str, typing.Any]
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        """
    def create_context(self, device_name: str, properties: collections.abc.Mapping[str, typing.Any]) -> RemoteContext:
        """
                    Creates a new remote shared context object on the specified accelerator device
                    using specified plugin-specific low-level device API parameters.
        
                    :param device_name: Name of a device to create a new shared context on.
                    :type device_name: str
                    :param properties: dict of device-specific shared context remote properties.
                    :type properties: dict[str, typing.Any]
                    :return: Remote context instance.
                    :rtype: openvino.RemoteContext
        """
    def get_available_devices(self) -> list[str]:
        """
                        Returns devices available for inference Core objects goes over all registered plugins.
        
                        GIL is released while running this function.
        
                        :returns: A list of devices. The devices are returned as: CPU, GPU.0, GPU.1, NPU...
                            If there more than one device of specific type, they are enumerated with .# suffix.
                            Such enumerated device can later be used as a device name in all Core methods like:
                            compile_model, query_model, set_property and so on.
                        :rtype: list[str]
        """
    def get_default_context(self, device_name: str) -> RemoteContext:
        """
                    Gets default (plugin-supplied) shared context object for the specified accelerator device.
        
                    :param device_name: Name of a device to get a default shared context from.
                    :type device_name: str
                    :return: Remote context instance.
                    :rtype: openvino.RemoteContext
        """
    @typing.overload
    def get_property(self, device_name: str, name: str, arguments: collections.abc.Mapping[str, typing.Any]) -> typing.Any:
        """
                    Gets properties dedicated to device behaviour.
        
                    :param device_name: A name of a device to get a properties value.
                    :type device_name: str
                    :param name: Property or name of Property.
                    :type name: str
                    :param arguments: Additional arguments to get a property.
                    :type arguments: dict[str, typing.Any]
                    :return: Extracted information from property.
                    :rtype: typing.Any
        """
    @typing.overload
    def get_property(self, device_name: str, property: str) -> typing.Any:
        """
                    Gets properties dedicated to device behaviour.
        
                    :param device_name: A name of a device to get a properties value.
                    :type device_name: str
                    :param property: Property or name of Property.
                    :type property: str
                    :return: Extracted information from property.
                    :rtype: typing.Any
        """
    @typing.overload
    def get_property(self, property: str) -> typing.Any:
        """
                    Gets properties dedicated to Core behaviour.
        
                    :param property: Property or name of Property.
                    :type property: str
                    :return: Extracted information from property.
                    :rtype: typing.Any
        """
    def get_versions(self, device_name: str) -> dict[str, Version]:
        """
                        Returns device plugins version information.
        
                        :param device_name: Device name to identify a plugin.
                        :type device_name: str
                        :return: Plugin version information.
                        :rtype: dict[str, openvino.Version]
        """
    def import_model(self, model_stream: typing.Any, device_name: str, properties: collections.abc.Mapping[str, typing.Any]) -> CompiledModel:
        """
                    Imports a compiled model from a previously exported one.
        
                    Advanced version of `import_model`. It utilizes, streams from standard
                    Python library `io`.
        
                    GIL is released while running this function.
        
        
                    :param model_stream: Input stream, containing a model previously exported, using export_model method.
                    :type model_stream: typing.Union[io.BytesIO, bytes]
                    :param device_name: Name of device to which compiled model is imported.
                                        Note: if device_name is not used to compile the original model, an exception is thrown.
                    :type device_name: str
                    :param properties: Optional map of pairs: (property name, property value) relevant only for this load operation.
                    :type properties: dict[str, typing.Any], optional
                    :return: A compiled model.
                    :rtype: openvino.CompiledModel
        
                    :Example:
                    .. code-block:: python
        
                        user_stream = io.BytesIO()
                        compiled.export_model(user_stream)
        
                        with open('./my_model', 'wb') as f:
                            f.write(user_stream.getvalue()) # or read() if seek(0) was applied before
        
                        # ...
        
                        new_compiled = core.import_model(user_stream, "CPU")
        """
    def query_model(self, model: Model, device_name: str, properties: collections.abc.Mapping[str, typing.Any] = {}) -> dict[str, str]:
        """
                    Query device if it supports specified model with specified properties.
        
                    GIL is released while running this function.
        
                    :param model: Model object to query.
                    :type model: openvino.Model
                    :param device_name: A name of a device to query.
                    :type device_name: str
                    :param properties: Optional dict of pairs: (property name, property value)
                    :type properties: dict[str, typing.Any]
                    :return: Pairs a operation name -> a device name supporting this operation.
                    :rtype: dict[str, str]
        """
    @typing.overload
    def read_model(self, model: bytes, weights: bytes = b'') -> Model:
        """
                    Reads models from IR / ONNX / PDPD / TF and TFLite formats.
        
                    GIL is released while running this function.
        
                    :param model: Bytes with model in IR / ONNX / PDPD / TF and TFLite format.
                    :type model: bytes
                    :param weights: Bytes with tensor's data.
                    :type weights: bytes
                    :return: A model.
                    :rtype: openvino.Model
        """
    @typing.overload
    def read_model(self, model: str, weights: str = '', config: collections.abc.Mapping[str, typing.Any] = {}) -> Model:
        """
                    Reads models from IR / ONNX / PDPD / TF and TFLite formats.
        
                    GIL is released while running this function.
        
                    :param model: A path to a model in IR / ONNX / PDPD / TF and TFLite format.
                    :type model: str
                    :param weights: A path to a data file For IR format (*.bin): if path is empty,
                                    it tries to read a bin file with the same name as xml and if the bin
                                    file with the same name was not found, loads IR without weights.
                                    For ONNX format (*.onnx): weights parameter is not used.
                                    For PDPD format (*.pdmodel) weights parameter is not used.
                                    For TF format (*.pb) weights parameter is not used.
                                    For TFLite format (*.tflite) weights parameter is not used.
                    :type weights: str
                    :param config: Optional map of pairs: (property name, property value) relevant only for this read operation.
                    :type config: dict[str, typing.Any], optional
                    :return: A model.
                    :rtype: openvino.Model
        """
    @typing.overload
    def read_model(self, model: str, weights: Tensor) -> Model:
        """
                    Reads models from IR / ONNX / PDPD / TF and TFLite formats.
        
                    GIL is released while running this function.
        
                    :param model: A string with model in IR / ONNX / PDPD / TF and TFLite format.
                    :type model: str
                    :param weights: Tensor with weights. Reading ONNX / PDPD / TF and TFLite models
                                    doesn't support loading weights from weights tensors.
                    :type weights: openvino.Tensor
                    :return: A model.
                    :rtype: openvino.Model
        """
    @typing.overload
    def read_model(self, model: typing.Any, weights: typing.Any = None, config: collections.abc.Mapping[str, typing.Any] = {}) -> Model:
        """
                    Reads models from IR / ONNX / PDPD / TF and TFLite formats.
        
                    GIL is released while running this function.
        
                    :param model: A path to a model in IR / ONNX / PDPD / TF and TFLite format or a model itself wrapped in io.ByesIO format.
                    :type model: typing.Union[pathlib.Path, io.BytesIO]
                    :param weights: A path to a data file For IR format (*.bin): if path is empty,
                                    it tries to read a bin file with the same name as xml and if the bin
                                    file with the same name was not found, loads IR without weights.
                                    For ONNX format (*.onnx): weights parameter is not used.
                                    For PDPD format (*.pdmodel) weights parameter is not used.
                                    For TF format (*.pb): weights parameter is not used.
                                    For TFLite format (*.tflite) weights parameter is not used.
                    :type weights: typing.Union[pathlib.Path, io.BytesIO]
                    :param config: Optional map of pairs: (property name, property value) relevant only for this read operation.
                    :type config: dict[str, typing.Any], optional
                    :return: A model.
                    :rtype: openvino.Model
        """
    @typing.overload
    def register_plugin(self, plugin_name: str, device_name: str) -> None:
        """
                        Register a new device and plugin which enable this device inside OpenVINO Runtime.
        
                        :param plugin_name: A path (absolute or relative) or name of a plugin. Depending on platform,
                                            `plugin_name` is wrapped with shared library suffix and prefix to identify
                                            library full name E.g. on Linux platform plugin name specified as `plugin_name`
                                            will be wrapped as `libplugin_name.so`.
                        :type plugin_name: str
                        :param device_name: A device name to register plugin for.
                        :type device_name: str
        """
    @typing.overload
    def register_plugin(self, plugin_name: str, device_name: str, config: collections.abc.Mapping[str, typing.Any]) -> None:
        """
                        Register a new device and plugin which enable this device inside OpenVINO Runtime.
        
                        :param plugin_name: A path (absolute or relative) or name of a plugin. Depending on platform,
                                            `plugin_name` is wrapped with shared library suffix and prefix to identify
                                            library full name E.g. on Linux platform plugin name specified as `plugin_name`
                                            will be wrapped as `libplugin_name.so`.
                        :type plugin_name: str
                        :param device_name: A device name to register plugin for.
                        :type device_name: str
                        :param config: Plugin default configuration
                        :type config: dict[str, typing.Any], optional
        """
    def register_plugins(self, xml_config_file: str) -> None:
        """
                        Registers a device plugin to OpenVINO Runtime Core instance using XML configuration
                        file with plugins description.
        
                        :param xml_config_file: A path to .xml file with plugins to register.
                        :type xml_config_file: str
        """
    @typing.overload
    def set_property(self, properties: collections.abc.Mapping[str, typing.Any]) -> None:
        """
                    Sets properties.
        
                    :param properties: dict of pairs: (property name, property value).
                    :type properties: dict[str, typing.Any]
        """
    @typing.overload
    def set_property(self, property: tuple[str, typing.Any]) -> None:
        """
                    Sets properties for the device.
        
                    :param property: tuple of (property name, matching property value).
                    :type property: tuple[str, typing.Any]
        """
    @typing.overload
    def set_property(self, device_name: str, properties: collections.abc.Mapping[str, typing.Any]) -> None:
        """
                    Sets properties for the device.
        
                    :param device_name: Name of the device.
                    :type device_name: str
                    :param properties: dict of pairs: (property name, property value).
                    :type properties: dict[str, typing.Any]
        """
    @typing.overload
    def set_property(self, device_name: str, property: tuple[str, typing.Any]) -> None:
        """
                    Sets properties for the device.
        
                    :param device_name: Name of the device.
                    :type device_name: str
                    :param property: tuple of (property name, matching property value).
                    :type property: tuple[str, typing.Any]
        """
    def unload_plugin(self, device_name: str) -> None:
        """
                        Unloads the previously loaded plugin identified by device_name from OpenVINO Runtime.
                        The method is needed to remove loaded plugin instance and free its resources.
                        If plugin for a specified device has not been created before, the method throws an exception.
        
                        :param device_name: A device name identifying plugin to remove from OpenVINO.
                        :type device_name: str
        """
    @property
    def available_devices(self) -> list[str]:
        """
                                            Returns devices available for inference Core objects goes over all registered plugins.
        
                                            GIL is released while running this function.
        
                                            :returns: A list of devices. The devices are returned as: CPU, GPU.0, GPU.1, NPU...
                                                If there more than one device of specific type, they are enumerated with .# suffix.
                                                Such enumerated device can later be used as a device name in all Core methods like:
                                                compile_model, query_model, set_property and so on.
                                            :rtype: list[str]
        """
class DecoderTransformationExtension(Extension):
    pass
class DescriptorTensor:
    """
    openvino.descriptor.Tensor wraps ov::descriptor::Tensor
    """
    def __repr__(self) -> str:
        ...
    def add_names(self, names: collections.abc.Set[str]) -> None:
        """
                        Adds names for tensor.
        
                        :param names: Add names.
                        :type names: set
        """
    def get_any_name(self) -> str:
        """
                        Returns any of set name.
        
                        :return: Any name.
                        :rtype: string
        """
    def get_element_type(self) -> Type:
        """
                        Returns the element type description.
        
                        :return: Type description.
                        :rtype: openvino.Type
        """
    def get_lower_value(self) -> Tensor:
        """
                        Returns the lower bound of the tensor.
        
                        :return: Lower bound.
                        :rtype: openvino.Tensor
        """
    def get_names(self) -> set[str]:
        """
                        Returns names.
        
                        :return: Get names.
                        :rtype: set
        """
    def get_partial_shape(self) -> PartialShape:
        """
                        Returns the partial shape description.
        
                        :return: PartialShape description.
                        :rtype: openvino.PartialShape
        """
    def get_rt_info(self) -> RTMap:
        """
                        Returns PyRTMap which is a dictionary of user defined runtime info.
        
                        :return: A dictionary of user defined data.
                        :rtype: openvino.RTMap
        """
    def get_shape(self) -> Shape:
        """
                        Returns the shape description.
        
                        :return: The shape description.
                        :rtype:  openvino.Shape
        """
    def get_upper_value(self) -> Tensor:
        """
                        Returns the upper bound of the tensor.
        
                        :return: Upper bound.
                        :rtype: openvino.Tensor
        """
    def get_value_symbol(self) -> list[Symbol]:
        """
                        Returns the list of symbols.
        
                        :return: list of Symbols.
                        :rtype: list[openvino.Symbol]
        """
    def set_lower_value(self, lower_bound: Tensor) -> None:
        """
                        Sets the lower bound of the tensor.
        
                        :param lower_bound: The lower bound value.
                        :type lower_bound: openvino.Tensor
        """
    def set_names(self, names: collections.abc.Set[str]) -> None:
        """
                        Set names for tensor.
        
                        :param names: set of names.
                        :type names: set
        """
    def set_upper_value(self, upper_bound: Tensor) -> None:
        """
                        Sets the upper bound of the tensor.
        
                        :param upper_bound: Sets the upper bound value.
                        :type upper_bound: openvino.Tensor
        """
    def set_value_symbol(self, value_symbol: collections.abc.Sequence[Symbol]) -> None:
        """
                        Sets the value symbol of the tensor.
        
                        :param value_symbol: list of Symbols
                        :type value_symbol: list[openvino.Symbol]
        """
    @property
    def any_name(self) -> str:
        ...
    @property
    def element_type(self) -> Type:
        ...
    @property
    def names(self) -> set[str]:
        ...
    @names.setter
    def names(self, arg1: collections.abc.Set[str]) -> None:
        ...
    @property
    def partial_shape(self) -> PartialShape:
        ...
    @property
    def rt_info(self) -> RTMap:
        ...
    @property
    def shape(self) -> Shape:
        ...
    @property
    def size(self) -> int:
        ...
class Dimension:
    """
    openvino.Dimension wraps ov::Dimension
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def dynamic() -> Dimension:
        ...
    @typing.overload
    def __eq__(self, arg0: Dimension) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: typing.SupportsInt) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, dimension: typing.SupportsInt) -> None:
        """
                        Construct a static dimension.
        
                        :param dimension: Value of the dimension.
                        :type dimension: int
        """
    @typing.overload
    def __init__(self, min_dimension: typing.SupportsInt, max_dimension: typing.SupportsInt) -> None:
        """
                        Construct a dynamic dimension with bounded range.
        
                        :param min_dimension: The lower inclusive limit for the dimension.
                        :type min_dimension: int
                        :param max_dimension: The upper inclusive limit for the dimension.
                        :type max_dimension: int
        """
    @typing.overload
    def __init__(self, str: str) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def compatible(self, dim: Dimension) -> bool:
        """
                        Check whether this dimension is capable of being merged
                        with the argument dimension.
        
                        :param dim: The dimension to compare this dimension with.
                        :type dim: Dimension
                        :return: True if this dimension is compatible with d, else False.
                        :rtype: bool
        """
    def get_length(self) -> int:
        """
                        Return this dimension as integer.
                        This dimension must be static and non-negative.
                        
                        :return: Value of the dimension.
                        :rtype: int
        """
    def get_max_length(self) -> int:
        """
                        Return this dimension's max_dimension as integer.
                        This dimension must be dynamic and non-negative.
        
                        :return: Value of the dimension.
                        :rtype: int
        """
    def get_min_length(self) -> int:
        """
                        Return this dimension's min_dimension as integer.
                        This dimension must be dynamic and non-negative.
        
                        :return: Value of the dimension.
                        :rtype: int
        """
    def get_symbol(self) -> Symbol:
        """
                        Return this dimension's symbol as Symbol object.
        
                        :return: Value of the dimension.
                        :rtype: openvino.Symbol
        """
    def has_symbol(self) -> bool:
        """
                      Check if Dimension has meaningful symbol.
        
                      :return: True if symbol was set, else False.
                      :rtype: bool
        """
    def refines(self, dim: Dimension) -> bool:
        """
                        Check whether this dimension is a refinement of the argument.
                        This dimension refines (or is a refinement of) d if:
        
                        (1) this and d are static and equal
                        (2) d dimension contains this dimension
        
                        this.refines(d) is equivalent to d.relaxes(this).
        
                        :param dim: The dimension to compare this dimension with.
                        :type dim: Dimension
                        :return: True if this dimension refines d, else False.
                        :rtype: bool
        """
    def relaxes(self, dim: Dimension) -> bool:
        """
                        Check whether this dimension is a relaxation of the argument.
                        This dimension relaxes (or is a relaxation of) d if:
        
                        (1) this and d are static and equal
                        (2) this dimension contains d dimension
        
                        this.relaxes(d) is equivalent to d.refines(this).
        
                        :param dim: The dimension to compare this dimension with.
                        :type dim: Dimension
                        :return: True if this dimension relaxes d, else False.
                        :rtype: bool
        """
    def same_scheme(self, dim: Dimension) -> bool:
        """
                        Return this dimension's max_dimension as integer.
                        This dimension must be dynamic and non-negative.
        
                        :param dim: The other dimension to compare this dimension to.
                        :type dim: Dimension
                        :return: True if this dimension and dim are both dynamic,
                                 or if they are both static and equal, otherwise False.
                        :rtype: bool
        """
    def set_symbol(self, symbol: Symbol) -> None:
        """
                        Sets provided Symbol as this dimension's symbol.
        
                        :param symbol: The symbol to set to this dimension.
                        :type symbol: openvino.Symbol
        """
    def to_string(self) -> str:
        ...
    @property
    def is_dynamic(self) -> bool:
        """
                                        Check if Dimension is dynamic.
                                        :return: True if dynamic, else False.
                                        :rtype: bool
        """
    @property
    def is_static(self) -> bool:
        """
                                        Check if Dimension is static.
        
                                        :return: True if static, else False.
                                        :rtype: bool
        """
    @property
    def max_length(self) -> int:
        """
                        Return this dimension's max_dimension as integer.
                        This dimension must be dynamic and non-negative.
        
                        :return: Value of the dimension.
                        :rtype: int
        """
    @property
    def min_length(self) -> int:
        """
                        Return this dimension's min_dimension as integer.
                        This dimension must be dynamic and non-negative.
        
                        :return: Value of the dimension.
                        :rtype: int
        """
class DiscreteTypeInfo:
    """
    openvino.DiscreteTypeInfo wraps ov::DiscreteTypeInfo
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: DiscreteTypeInfo) -> bool:
        ...
    def __ge__(self, arg0: DiscreteTypeInfo) -> bool:
        ...
    def __gt__(self, arg0: DiscreteTypeInfo) -> bool:
        ...
    def __init__(self, name: str, version_id: str) -> None:
        ...
    def __le__(self, arg0: DiscreteTypeInfo) -> bool:
        ...
    def __lt__(self, arg0: DiscreteTypeInfo) -> bool:
        ...
    def __ne__(self, arg0: DiscreteTypeInfo) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def hash(self) -> int:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def parent(self) -> DiscreteTypeInfo:
        ...
    @property
    def version_id(self) -> str:
        ...
class Extension:
    """
    openvino.Extension provides the base interface for OpenVINO extensions.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class FrontEnd:
    """
    openvino.frontend.FrontEnd wraps ov::frontend::FrontEnd
    """
    def __init__(self, other: FrontEnd) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def add_extension(self, arg0: Extension) -> None:
        """
                        Add extension defined by an object inheriting from Extension
                        used in order to extend capabilities of Frontend.
        
                        :param extension: Provided extension object.
                        :type extension: Extension
        """
    @typing.overload
    def add_extension(self, arg0: collections.abc.Sequence[Extension]) -> None:
        """
                        Add extensions defined by objects inheriting from Extension
                        used in order to extend capabilities of Frontend.
        
                        :param extension: Provided extension objects.
                        :type extension: list[Extension]
        """
    @typing.overload
    def add_extension(self, arg0: typing.Any) -> None:
        """
                        Add extension defined in external library indicated by a extension_path
                        used in order to extend capabilities of Frontend.
        
                        :param extension_path: A path to extension.
                        :type extension_path: str, Path
        """
    @typing.overload
    def convert(self, model: InputModel) -> Model:
        """
                        Completely convert and normalize entire function, throws if it is not possible.
        
                        :param model: Input model.
                        :type model: openvino.frontend.InputModel
                        :return: Fully converted OpenVINO Model.
                        :rtype: openvino.Model
        """
    @typing.overload
    def convert(self, model: typing.Any) -> None:
        """
                        Completely convert the remaining, not converted part of a function.
        
                        :param model: Partially converted OpenVINO model.
                        :type model: openvino.frontend.Model
                        :return: Fully converted OpenVINO Model.
                        :rtype: openvino.Model
        """
    def convert_partially(self, model: InputModel) -> Model:
        """
                        Convert only those parts of the model that can be converted leaving others as-is.
                        Converted parts are not normalized by additional transformations; normalize function or
                        another form of convert function should be called to finalize the conversion process.
        
                        :param model : Input model.
                        :type model: openvino.frontend.InputModel
                        :return: Partially converted OpenVINO Model.
                        :rtype: openvino.Model
        """
    def decode(self, model: InputModel) -> Model:
        """
                        Convert operations with one-to-one mapping with decoding nodes.
                        Each decoding node is an nGraph node representing a single FW operation node with
                        all attributes represented in FW-independent way.
        
                        :param model : Input model.
                        :type model: openvino.frontend.InputModel
                        :return: OpenVINO Model after decoding.
                        :rtype: openvino.Model
        """
    def get_name(self) -> str:
        """
                        Gets name of this FrontEnd. Can be used by clients
                        if frontend is selected automatically by FrontEndManager::load_by_model.
        
                        :return: Current frontend name. Returns empty string if not implemented.
                        :rtype: str
        """
    def load(self, path: typing.Any, enable_mmap: bool = True) -> InputModel:
        """
                        Loads an input model.
        
                        :param path: Object describing the model. It can be path to model file.
                        :type path: Any
                        :param enable_mmap: Use mmap feature to map memory of a model's weights instead of reading directly. Optional. The default value is true.
                        :type enable_mmap: boolean
                        :return: Loaded input model.
                        :rtype: openvino.frontend.InputModel
        """
    def normalize(self, model: typing.Any) -> None:
        """
                        Runs normalization passes on function that was loaded with partial conversion.
        
                        :param model : Partially converted OpenVINO model.
                        :type model: openvino.Model
        """
    def supported(self, model: typing.Any) -> bool:
        """
                        Checks if model type is supported.
        
                        :param model: Object describing the model. It can be path to model file.
                        :type model: Any
                        :return: True if model type is supported, otherwise False.
                        :rtype: bool
        """
class FrontEndManager:
    """
    openvino.frontend.FrontEndManager wraps ov::frontend::FrontEndManager
    """
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def get_available_front_ends(self) -> list[str]:
        """
                        Gets list of registered frontends.
        
                        :return: list of available frontend names.
                        :rtype: list[str]
        """
    def load_by_framework(self, framework: str) -> FrontEnd:
        """
                        Loads frontend by name of framework and capabilities.
        
                        :param framework: Framework name. Throws exception if name is not in list of available frontends.
                        :type framework: str
                        :return: Frontend interface for further loading of models.
                        :rtype: openvino.frontend.FrontEnd
        """
    def load_by_model(self, model: typing.Any) -> FrontEnd:
        """
                        Selects and loads appropriate frontend depending on model type or model file extension and other file info (header).
        
                        :param model_path: A model object or path to a model file/directory.
                        :type model_path: Any
                        :return: Frontend interface for further loading of models. 'None' if no suitable frontend is found.
                        :rtype: openvino.frontend.FrontEnd
        """
    def register_front_end(self, name: str, library_path: str) -> None:
        """
                        Register frontend with name and factory loaded from provided library.
        
                        :param name: Name of front end.
                        :type name: str
        
                        :param library_path: Path (absolute or relative) or name of a frontend library. If name is
                        provided, depending on platform, it will be wrapped with shared library suffix and prefix
                        to identify library full name.
                        :type library_path: str
        
                        :return: None
        """
class GeneralFailure(Exception):
    pass
class InferRequest:
    """
    openvino.InferRequest represents infer request which can be run in asynchronous or synchronous manners.
    """
    def __init__(self, other: InferRequest) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def cancel(self) -> None:
        """
                    Cancels inference request.
        """
    def get_compiled_model(self) -> CompiledModel:
        """
                    Returns the compiled model.
        
                    :return: Compiled model object.
                    :rtype: openvino.CompiledModel
        """
    @typing.overload
    def get_input_tensor(self, index: typing.SupportsInt) -> Tensor:
        """
                    Gets input tensor of InferRequest.
        
                    :param idx: An index of tensor to get.
                    :type idx: int
                    :return: An input Tensor with index idx for the model.
                             If a tensor with specified idx is not found,
                    an exception is thrown.
                    :rtype: openvino.Tensor
        """
    @typing.overload
    def get_input_tensor(self) -> Tensor:
        """
                    Gets input tensor of InferRequest.
        
                    :return: An input Tensor for the model.
                             If model has several inputs, an exception is thrown.
                    :rtype: openvino.Tensor
        """
    @typing.overload
    def get_output_tensor(self, index: typing.SupportsInt) -> Tensor:
        """
                    Gets output tensor of InferRequest.
        
                    :param idx: An index of tensor to get.
                    :type idx: int
                    :return: An output Tensor with index idx for the model.
                             If a tensor with specified idx is not found, an exception is thrown.
                    :rtype: openvino.Tensor
        """
    @typing.overload
    def get_output_tensor(self) -> Tensor:
        """
                    Gets output tensor of InferRequest.
        
                    :return: An output Tensor for the model.
                             If model has several outputs, an exception is thrown.
                    :rtype: openvino.Tensor
        """
    def get_profiling_info(self) -> list[ProfilingInfo]:
        """
                    Queries performance is measured per layer to get feedback on what
                    is the most time-consuming operation, not all plugins provide
                    meaningful data.
        
                    GIL is released while running this function.
        
                    :return: list of profiling information for operations in model.
                    :rtype: list[openvino.ProfilingInfo]
        """
    @typing.overload
    def get_tensor(self, name: str) -> Tensor:
        """
                    Gets input/output tensor of InferRequest.
        
                    :param name: Name of tensor to get.
                    :type name: str
                    :return: A Tensor object with given name.
                    :rtype: openvino.Tensor
        """
    @typing.overload
    def get_tensor(self, port: ConstOutput) -> Tensor:
        """
                    Gets input/output tensor of InferRequest.
        
                    :param port: Port of tensor to get.
                    :type port: openvino.ConstOutput
                    :return: A Tensor object for the port.
                    :rtype: openvino.Tensor
        """
    @typing.overload
    def get_tensor(self, port: Output) -> Tensor:
        """
                    Gets input/output tensor of InferRequest.
        
                    :param port: Port of tensor to get.
                    :type port: openvino.Output
                    :return: A Tensor object for the port.
                    :rtype: openvino.Tensor
        """
    @typing.overload
    def infer(self, inputs: Tensor, share_outputs: bool, decode_strings: bool) -> typing.Any:
        """
                    Infers specified input(s) in synchronous mode.
                    Blocks all methods of InferRequest while request is running.
                    Calling any method will lead to throwing exceptions.
        
                    GIL is released while running the inference.
        
                    :param inputs: Data to set on single input tensor.
                    :type inputs: openvino.Tensor
                    :return: Dictionary of results from output tensors with ports as keys.
                    :rtype: dict[openvino.ConstOutput, numpy.array]
        """
    @typing.overload
    def infer(self, inputs: dict, share_outputs: bool, decode_strings: bool) -> typing.Any:
        """
                    Infers specified input(s) in synchronous mode.
                    Blocks all methods of InferRequest while request is running.
                    Calling any method will lead to throwing exceptions.
        
                    GIL is released while running the inference.
        
                    :param inputs: Data to set on input tensors.
                    :type inputs: dict[Union[int, str, openvino.ConstOutput], openvino.Tensor]
                    :return: Dictionary of results from output tensors with ports as keys.
                    :rtype: dict[openvino.ConstOutput, numpy.array]
        """
    def query_state(self) -> list[VariableState]:
        """
                    Gets state control interface for given infer request.
        
                    GIL is released while running this function.
        
                    :return: list of VariableState objects.
                    :rtype: list[openvino.VariableState]
        """
    def reset_state(self) -> None:
        """
                    Resets all internal variable states for relevant infer request to
                    a value specified as default for the corresponding `ReadValue` node
        """
    def set_callback(self, callback: collections.abc.Callable, userdata: typing.Any) -> None:
        """
                    Sets a callback function that will be called on success or failure of asynchronous InferRequest.
        
                    :param callback: Function defined in Python.
                    :type callback: function
                    :param userdata: Any data that will be passed inside callback call.
                    :type userdata: Any
        """
    @typing.overload
    def set_input_tensor(self, index: typing.SupportsInt, tensor: Tensor) -> None:
        """
                    Sets input tensor of InferRequest.
        
                    :param idx: Index of input tensor. If idx is greater than number of model's inputs,
                                an exception is thrown.
                    :type idx: int
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's input element_type and shape.
                    :type tensor: openvino.Tensor
        """
    @typing.overload
    def set_input_tensor(self, tensor: Tensor) -> None:
        """
                    Sets input tensor of InferRequest with single input.
                    If model has several inputs, an exception is thrown.
        
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's input element_type and shape.
                    :type tensor: openvino.Tensor
        """
    @typing.overload
    def set_input_tensors(self, inputs: dict) -> None:
        """
                    Set input tensors using given indexes.
        
                    :param inputs: Data to set on output tensors.
                    :type inputs: dict[int, openvino.Tensor]
        """
    @typing.overload
    def set_input_tensors(self, tensors: collections.abc.Sequence[Tensor]) -> None:
        """
                    Sets batch of tensors for single input data.
                    Model input needs to have batch dimension and the number of `tensors`
                    needs to match with batch size.
        
                    :param tensors:  Input tensors for batched infer request. The type of each tensor
                                     must match the model input element type and shape (except batch dimension).
                                     Total size of tensors needs to match with input's size.
                    :type tensors: list[openvino.Tensor]
        """
    @typing.overload
    def set_input_tensors(self, idx: typing.SupportsInt, tensors: collections.abc.Sequence[Tensor]) -> None:
        """
                    Sets batch of tensors for single input data to infer by index.
                    Model input needs to have batch dimension and the number of `tensors`
                    needs to match with batch size.
        
                    :param idx: Index of input tensor.
                    :type idx: int
                    :param tensors: Input tensors for batched infer request. The type of each tensor
                                    must match the model input element type and shape (except batch dimension).
                                    Total size of tensors needs to match with input's size.
        """
    @typing.overload
    def set_output_tensor(self, index: typing.SupportsInt, tensor: Tensor) -> None:
        """
                    Sets output tensor of InferRequest.
        
                    :param idx: Index of output tensor.
                    :type idx: int
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's output element_type and shape.
                    :type tensor: openvino.Tensor
        """
    @typing.overload
    def set_output_tensor(self, tensor: Tensor) -> None:
        """
                    Sets output tensor of InferRequest with single output.
                    If model has several outputs, an exception is thrown.
        
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's output element_type and shape.
                    :type tensor: openvino.Tensor
        """
    def set_output_tensors(self, outputs: dict) -> None:
        """
                    Set output tensors using given indexes.
        
                    :param outputs: Data to set on output tensors.
                    :type outputs: dict[int, openvino.Tensor]
        """
    @typing.overload
    def set_tensor(self, name: str, tensor: RemoteTensor) -> None:
        """
                    Sets input/output tensor of InferRequest.
        
                    :param name: Name of input/output tensor.
                    :type name: str
                    :param tensor: RemoteTensor object. The element_type and shape of a tensor
                                   must match the model's input/output element_type and shape.
                    :type tensor: openvino.RemoteTensor
        """
    @typing.overload
    def set_tensor(self, name: str, tensor: Tensor) -> None:
        """
                    Sets input/output tensor of InferRequest.
        
                    :param name: Name of input/output tensor.
                    :type name: str
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's input/output element_type and shape.
                    :type tensor: openvino.Tensor
        """
    @typing.overload
    def set_tensor(self, port: ConstOutput, tensor: Tensor) -> None:
        """
                    Sets input/output tensor of InferRequest.
        
                    :param port: Port of input/output tensor.
                    :type port: openvino.ConstOutput
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's input/output element_type and shape.
                    :type tensor: openvino.Tensor
        """
    @typing.overload
    def set_tensor(self, port: Output, tensor: Tensor) -> None:
        """
                    Sets input/output tensor of InferRequest.
        
                    :param port: Port of input/output tensor.
                    :type port: openvino.Output
                    :param tensor: Tensor object. The element_type and shape of a tensor
                                   must match the model's input/output element_type and shape.
                    :type tensor: openvino.Tensor
        """
    @typing.overload
    def set_tensors(self, inputs: dict) -> None:
        """
                    Set tensors using given keys.
        
                    :param inputs: Data to set on tensors.
                    :type inputs: dict[Union[int, str, openvino.ConstOutput], openvino.Tensor]
        """
    @typing.overload
    def set_tensors(self, tensor_name: str, tensors: collections.abc.Sequence[Tensor]) -> None:
        """
                    Sets batch of tensors for input data to infer by tensor name.
                    Model input needs to have batch dimension and the number of tensors needs to be
                    matched with batch size. Current version supports set tensors to model inputs only.
                    In case if `tensor_name` is associated with output (or any other non-input node),
                    an exception will be thrown.
        
                    :param tensor_name: Name of input tensor.
                    :type tensor_name: str
                    :param tensors: Input tensors for batched infer request. The type of each tensor
                                    must match the model input element type and shape (except batch dimension).
                                    Total size of tensors needs to match with input's size.
                    :type tensors: list[openvino.Tensor]
        """
    @typing.overload
    def set_tensors(self, port: ConstOutput, tensors: collections.abc.Sequence[Tensor]) -> None:
        """
                    Sets batch of tensors for input data to infer by tensor name.
                    Model input needs to have batch dimension and the number of tensors needs to be
                    matched with batch size. Current version supports set tensors to model inputs only.
                    In case if `port` is associated with output (or any other non-input node),
                    an exception will be thrown.
        
        
                    :param port: Port of input tensor.
                    :type port: openvino.ConstOutput
                    :param tensors: Input tensors for batched infer request. The type of each tensor
                                    must match the model input element type and shape (except batch dimension).
                                    Total size of tensors needs to match with input's size.
                    :type tensors: list[openvino.Tensor]
                    :rtype: None
        """
    @typing.overload
    def start_async(self, inputs: Tensor, userdata: typing.Any) -> None:
        """
                    Starts inference of specified input(s) in asynchronous mode.
                    Returns immediately. Inference starts also immediately.
        
                    GIL is released while running the inference.
        
                    Calling any method on this InferRequest while the request is
                    running will lead to throwing exceptions.
        
                    :param inputs: Data to set on single input tensors.
                    :type inputs: openvino.Tensor
                    :param userdata: Any data that will be passed inside callback call.
                    :type userdata: Any
        """
    @typing.overload
    def start_async(self, inputs: dict, userdata: typing.Any) -> None:
        """
                    Starts inference of specified input(s) in asynchronous mode.
                    Returns immediately. Inference starts also immediately.
        
                    GIL is released while running the inference.
        
                    Calling any method on this InferRequest while the request is
                    running will lead to throwing exceptions.
        
                    :param inputs: Data to set on input tensors.
                    :type inputs: dict[Union[int, str, openvino.ConstOutput], openvino.Tensor]
                    :param userdata: Any data that will be passed inside callback call.
                    :type userdata: Any
        """
    def wait(self) -> None:
        """
                    Waits for the result to become available.
                    Blocks until the result becomes available.
        
                    GIL is released while running this function.
        """
    def wait_for(self, timeout: typing.SupportsInt) -> bool:
        """
                    Waits for the result to become available.
                    Blocks until specified timeout has elapsed or
                    the result becomes available, whichever comes first.
        
                    GIL is released while running this function.
        
                    :param timeout: Maximum duration in milliseconds (ms) of blocking call.
                    :type timeout: int
                    :return: True if InferRequest is ready, False otherwise.
                    :rtype: bool
        """
    @property
    def input_tensors(self) -> list[Tensor]:
        """
                                        Gets all input tensors of this InferRequest.
                                        
                                        :rtype: list[openvino.Tensor]
        """
    @property
    def latency(self) -> float:
        """
                    Gets latency of this InferRequest.
        
                    :rtype: float
        """
    @property
    def model_inputs(self) -> list[ConstOutput]:
        """
                    Gets all inputs of a compiled model which was used to create this InferRequest.
        
                    :rtype: list[openvino.ConstOutput]
        """
    @property
    def model_outputs(self) -> list[ConstOutput]:
        """
                    Gets all outputs of a compiled model which was used to create this InferRequest.
        
                    :rtype: list[openvino.ConstOutput]
        """
    @property
    def output_tensors(self) -> list[Tensor]:
        """
                                        Gets all output tensors of this InferRequest.
                                        
                                        :rtype: list[openvino.Tensor]
        """
    @property
    def profiling_info(self) -> list[ProfilingInfo]:
        """
                    Performance is measured per layer to get feedback on the most time-consuming operation.
                    Not all plugins provide meaningful data!
        
                    GIL is released while running this function.
                    
                    :return: Inference time.
                    :rtype: list[openvino.ProfilingInfo]
        """
    @property
    def results(self) -> dict:
        """
                    Gets all outputs tensors of this InferRequest.
        
                    Note: All string-based data is decoded by default.
        
                    :return: Dictionary of results from output tensors with ports as keys.
                    :rtype: dict[openvino.ConstOutput, numpy.array]
        """
    @property
    def userdata(self) -> typing.Any:
        """
                    Gets currently held userdata.
        
                    :rtype: Any
        """
class InitializationFailure(Exception):
    pass
class Input:
    """
    openvino.Input wraps ov::Input<Node>
    """
    def __repr__(self) -> str:
        ...
    def get_element_type(self) -> Type:
        """
                        The element type of the input referred to by this input handle.
        
                        :return: Type of the input.
                        :rtype: openvino.Type
        """
    def get_index(self) -> int:
        """
                        The index of the input referred to by this input handle.
        
                        :return: Index value as integer.
                        :rtype: int
        """
    def get_node(self) -> ...:
        """
                        Get node referenced by this input handle.
        
                        :return: Node object referenced by this input handle.
                        :rtype: openvino.Node
        """
    def get_partial_shape(self) -> PartialShape:
        """
                        The partial shape of the input referred to by this input handle.
        
                        :return: PartialShape of the input.
                        :rtype: openvino.PartialShape
        """
    def get_rt_info(self) -> RTMap:
        """
                        Returns RTMap which is a dictionary of user defined runtime info.
        
                        :return: A dictionary of user defined data.
                        :rtype: openvino.RTMap
        """
    def get_shape(self) -> Shape:
        """
                        The shape of the input referred to by this input handle.
        
                        :return: Shape of the input.
                        :rtype: openvino.Shape
        """
    def get_source_output(self) -> Output:
        """
                        A handle to the output that is connected to this input.
        
                        :return: Output that is connected to the input.
                        :rtype: openvino.Output
        """
    def get_tensor(self) -> DescriptorTensor:
        """
                        A reference to the tensor descriptor for this input.
        
                        :return: Tensor of the input.
                        :rtype: openvino._pyopenvino.DescriptorTensor
        """
    def replace_source_output(self, new_source_output: Output) -> None:
        """
                        Replaces the source output of this input.
        
                        :param new_source_output: A handle for the output that will replace this input's source.
                        :type new_source_output: openvino.Input
        """
    def set_rt_info(self, value: typing.Any, key: str) -> None:
        """
                        Add a value to the runtime info.
        
                        :param value: Value for the runtime info.
                        :type value: Any
                        :param key: String that defines a key in the runtime info dictionary.
                        :type key: str
        """
    @property
    def rt_info(self) -> RTMap:
        ...
class InputModel:
    """
    openvino.frontend.InputModel wraps ov::frontend::InputModel
    """
    def add_name_for_tensor(self, tensor: Place, new_name: str) -> None:
        """
                        Adds new name for tensor
        
                        :param tensor: Tensor place.
                        :type tensor: openvino.frontend.Place
                        :param new_name: New name to be added to this place.
                        :type new_name: str
        """
    def add_output(self, place: Place) -> Place:
        """
                        Assign this place as new output or add necessary nodes to represent a new output.
        
                        :param place: Anchor point to add an output.
                        :type place: openvino.frontend.Place
        """
    def cut_and_add_new_input(self, place: Place, new_name: str = '') -> None:
        """
                        Cut immediately before this place and assign this place as new input; prune
                        all nodes that don't contribute to any output.
        
                       :param place: New place to be assigned as input.
                       :type place: openvino.frontend.Place
                       :param new_name: Optional new name assigned to this input place.
                       :type new_name: str
        """
    def cut_and_add_new_output(self, place: Place, new_name: str = '') -> None:
        """
                        Cut immediately before this place and assign this place as new output; prune
                        all nodes that don't contribute to any output.
        
                        :param place: New place to be assigned as output.
                        :type place: openvino.frontend.Place
                        :param new_name: Optional new name assigned to this output place.
                        :type new_name: str
        """
    def extract_subgraph(self, inputs: collections.abc.Sequence[Place], outputs: collections.abc.Sequence[Place]) -> None:
        """
                        Leaves only subgraph that are defined by new inputs and new outputs.
        
                        :param inputs: Array of new input places.
                        :type inputs: list[openvino.frontend.Place]
                        :param outputs: Array of new output places.
                        :type outputs: list[openvino.frontend.Place]
        """
    def free_name_for_operation(self, name: str) -> None:
        """
                        Unassign specified name from operation place(s).
        
                        :param name: Name of operation.
                        :type name: str
        """
    def free_name_for_tensor(self, name: str) -> None:
        """
                        Unassign specified name from tensor place(s).
        
                        :param name: Name of tensor.
                        :type name: str
        """
    def get_element_type(self, place: Place) -> Type:
        """
                        Returns current element type used for this place.
        
                        :param place: Model place.
                        :type place: openvino.frontend.Place
                        :return: Element type for this place.
                        :rtype: openvino.Type
        """
    def get_inputs(self) -> list[Place]:
        """
                        Returns all inputs for a model.
        
                        :return: A list of input places.
                        :rtype: list[openvino.frontend.Place]
        """
    def get_outputs(self) -> list[Place]:
        """
                        Returns all outputs for a model. An output is a terminal place in a graph where data escapes the flow.
        
                        :return: A list of output places.
                        :rtype: list[openvino.frontend.Place]
        """
    def get_partial_shape(self, place: Place) -> PartialShape:
        """
                        Returns current partial shape used for this place.
        
                        :param place: Model place.
                        :type place: openvino.frontend.Place
                        :return: Partial shape for this place.
                        :rtype: openvino.PartialShape
        """
    def get_place_by_input_index(self, input_idx: typing.SupportsInt) -> Place:
        """
                        Returns a tensor place by an input index.
        
                        :param input_idx: Index of model input.
                        :type input_idx: int
                        :return: Tensor place corresponding to specified input index or nullptr.
                        :rtype: openvino.frontend.Place
        """
    def get_place_by_operation_name(self, operation_name: str) -> Place:
        """
                        Returns an operation place by an operation name following framework conventions, or
                        nullptr if an operation with this name doesn't exist.
        
                        :param operation_name: Name of operation.
                        :type operation_name: str
                        :return: Place representing operation.
                        :rtype: openvino.frontend.Place
        """
    def get_place_by_operation_name_and_input_port(self, operation_name: str, input_port_index: typing.SupportsInt) -> Place:
        """
                        Returns an input port place by operation name and appropriate port index.
        
                        :param operation_name: Name of operation.
                        :type operation_name: str
                        :param input_port_index: Index of input port for this operation.
                        :type input_port_index: int
                        :return: Place representing input port of operation.
                        :rtype: openvino.frontend.Place
        """
    def get_place_by_operation_name_and_output_port(self, operation_name: str, output_port_index: typing.SupportsInt) -> Place:
        """
                        Returns an output port place by operation name and appropriate port index.
        
                        :param operation_name: Name of operation.
                        :type operation_name: str
                        :param output_port_index: Index of output port for this operation.
                        :type output_port_index: int
                        :return: Place representing output port of operation.
                        :rtype: openvino.frontend.Place
        """
    def get_place_by_tensor_name(self, tensor_name: str) -> Place:
        """
                        Returns a tensor place by a tensor name following framework conventions, or
                        nullptr if a tensor with this name doesn't exist.
        
                        :param tensor_name: Name of tensor.
                        :type tensor_name: str
                        :return: Tensor place corresponding to specified tensor name.
                        :rtype: openvino.frontend.Place
        """
    def override_all_inputs(self, inputs: collections.abc.Sequence[Place]) -> None:
        """
                        Modifies the graph to use new inputs instead of existing ones. New inputs
                        should completely satisfy all existing outputs.
        
                        :param inputs: Array of new input places.
                        :type inputs: list[openvino.frontend.Place]
        """
    def override_all_outputs(self, outputs: collections.abc.Sequence[Place]) -> None:
        """
                        Replaces all existing outputs with new ones removing all data flow that
                        is not required for new outputs.
        
                        :param outputs: Vector with places that will become new outputs; may intersect existing outputs.
                        :type outputs: list[openvino.frontend.Place]
        """
    def remove_output(self, place: Place) -> None:
        """
                        Removes any sinks directly attached to this place with all inbound data flow
                        if it is not required by any other output.
        
                        :param place: Model place.
                        :type place: openvino.frontend.Place
        """
    def set_element_type(self, place: Place, type: Type) -> None:
        """
                        Sets new element type for a place.
        
                        :param place: Model place.
                        :type place: openvino.frontend.Place
                        :param type: New element type.
                        :type type: openvino.Type
        """
    def set_name_for_dimension(self, place: Place, dim_index: typing.SupportsInt, dim_name: str) -> None:
        """
                        Set name for a particular dimension of a place (e.g. batch dimension).
        
                        :param place: Model's place.
                        :type place: openvino.frontend.Place
                        :param dim_index: Dimension index.
                        :type dim_index: int
                        :param dim_name: Name to assign on this dimension.
                        :type dum_name: str
        """
    def set_name_for_operation(self, operation: Place, new_name: str) -> None:
        """
                        Adds new name for tensor.
        
                        :param operation: Operation place.
                        :type operation: openvino.frontend.Place
                        :param new_name: New name for this operation.
                        :type new_name: str
        """
    def set_name_for_tensor(self, tensor: Place, new_name: str) -> None:
        """
                        Sets name for tensor. Overwrites existing names of this place.
        
                        :param tensor: Tensor place.
                        :type tensor: openvino.frontend.Place
                        :param new_name: New name for this tensor.
                        :type new_name: str
        """
    def set_partial_shape(self, place: Place, shape: PartialShape) -> None:
        """
                        Defines all possible shape that may be used for this place; place should be
                        uniquely refer to some data. This partial shape will be converted to corresponding
                        shape of results ngraph nodes and will define shape inference when the model is
                        converted to ngraph.
        
                        :param place: Model place.
                        :type place: openvino.frontend.Place
                        :param shape: Partial shape for this place.
                        :type shape: openvino.PartialShape
        """
    def set_tensor_value(self, place: Place, value: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]) -> None:
        """
                    Sets new element type for a place.
        
                    :param place: Model place.
                    :type place: openvino.frontend.Place
                    :param value: New value to assign.
                    :type value: numpy.ndarray
        """
class Iterator:
    def __iter__(self: typing.Iterator) -> typing.Iterator:
        ...
    def __next__(self: typing.Iterator) -> typing.Any:
        ...
class Layout:
    """
    openvino.Layout wraps ov::Layout
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def scalar() -> Layout:
        ...
    @typing.overload
    def __eq__(self, arg0: Layout) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: str) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, layout_str: str) -> None:
        ...
    @typing.overload
    def __ne__(self, arg0: Layout) -> bool:
        ...
    @typing.overload
    def __ne__(self, arg0: str) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_index_by_name(self, dimension_name: str) -> int:
        ...
    def has_name(self, dimension_name: str) -> bool:
        ...
    def to_string(self) -> str:
        ...
    @property
    def empty(self) -> bool:
        ...
class Model:
    """
    openvino.Model wraps ov::Model
    """
    friendly_name: str
    @typing.overload
    def __init__(self, other: Model) -> None:
        ...
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[op.Result], sinks: collections.abc.Sequence[Node], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                            Create user-defined Model which is a representation of a model.
        
                            :param results: list of results.
                            :type results: list[op.Result]
                            :param sinks: list of Nodes to be used as Sinks (e.g. Assign ops).
                            :type sinks: list[openvino.Node]
                            :param parameters: list of parameters.
                            :type parameters: list[op.Parameter]
                            :param name: String to set as model's friendly name.
                            :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[op.Result], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                            Create user-defined Model which is a representation of a model.
        
                            :param results: list of results.
                            :type results: list[op.Result]
                            :param parameters: list of parameters.
                            :type parameters: list[op.Parameter]
                            :param name: String to set as model's friendly name.
                            :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Node], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model.
        
                    :param results: list of Nodes to be used as results.
                    :type results: list[openvino.Node]
                    :param parameters: list of parameters.
                    :type parameters:  list[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, result: Node, parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                            Create user-defined Model which is a representation of a model.
        
                            :param result: Node to be used as result.
                            :type result: openvino.Node
                            :param parameters: list of parameters.
                            :type parameters: list[op.Parameter]
                            :param name: String to set as model's friendly name.
                            :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Output], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of outputs.
                    :type results: list[openvino.Output]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Output], sinks: collections.abc.Sequence[Node], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of outputs.
                    :type results: list[openvino.Output]
                    :param sinks: list of Nodes to be used as Sinks (e.g. Assign ops).
                    :type sinks: list[openvino.Node]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Output], sinks: collections.abc.Sequence[Output], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of outputs.
                    :type results: list[openvino.Output]
                    :param sinks: list of Output sink node handles.
                    :type sinks: list[openvino.Output]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Output], sinks: collections.abc.Sequence[Output], parameters: collections.abc.Sequence[op.Parameter], variables: collections.abc.Sequence[op.util.Variable], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of outputs.
                    :type results: list[openvino.Output]
                    :param sinks: list of Output sink node handles.
                    :type sinks: list[openvino.Output]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param variables: list of variables.
                    :type variables: list[op.util.Variable]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[op.Result], sinks: collections.abc.Sequence[Output], parameters: collections.abc.Sequence[op.Parameter], name: str = '') -> None:
        """
                Create user-defined Model which is a representation of a model
        
                :param results: list of results.
                :type results: list[op.Result]
                :param sinks: list of Output sink node handles.
                :type sinks: list[openvino.Output]
                :param parameters: list of parameters.
                :type parameters: list[op.Parameter]
                :param name: String to set as model's friendly name.
                :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[op.Result], sinks: collections.abc.Sequence[Output], parameters: collections.abc.Sequence[op.Parameter], variables: collections.abc.Sequence[op.util.Variable], name: str = '') -> None:
        """
                Create user-defined Model which is a representation of a model
        
                :param results: list of results.
                :type results: list[op.Result]
                :param sinks: list of Output sink node handles.
                :type sinks: list[openvino.Output]
                :param parameters: list of parameters.
                :type parameters: list[op.Parameter]
                :param variables: list of variables.
                :type variables: list[op.util.Variable]
                :param name: String to set as model's friendly name.
                :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[op.Result], sinks: collections.abc.Sequence[Node], parameters: collections.abc.Sequence[op.Parameter], variables: collections.abc.Sequence[op.util.Variable], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of results.
                    :type results: list[op.Result]
                    :param sinks: list of Nodes to be used as Sinks (e.g. Assign ops).
                    :type sinks: list[openvino.Node]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param variables: list of variables.
                    :type variables: list[op.util.Variable]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Output], sinks: collections.abc.Sequence[Node], parameters: collections.abc.Sequence[op.Parameter], variables: collections.abc.Sequence[op.util.Variable], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of results.
                    :type results: list[openvino.Output]
                    :param sinks: list of Nodes to be used as Sinks (e.g. Assign ops).
                    :type sinks: list[openvino.Node]
                    :param variables: list of variables.
                    :type variables: list[op.util.Variable]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[op.Result], parameters: collections.abc.Sequence[op.Parameter], variables: collections.abc.Sequence[op.util.Variable], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of results.
                    :type results: list[op.Result]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param variables: list of variables.
                    :type variables: list[op.util.Variable]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    @typing.overload
    def __init__(self, results: collections.abc.Sequence[Output], parameters: collections.abc.Sequence[op.Parameter], variables: collections.abc.Sequence[op.util.Variable], name: str = '') -> None:
        """
                    Create user-defined Model which is a representation of a model
        
                    :param results: list of results.
                    :type results: list[openvino.Output]
                    :param parameters: list of parameters.
                    :type parameters: list[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
        """
    def __repr__(self) -> str:
        ...
    def _get_raw_address(self) -> int:
        """
                Returns a raw address of the Model object from C++.
        
                Use this function in order to compare underlying C++ addresses instead of using `__eq__` in Python.
        
                :return: a raw address of the Model object.
                :rtype: int
        """
    def add_outputs(self, outputs: typing.Any) -> list[Output]:
        ...
    def add_parameters(self, parameters: collections.abc.Sequence[op.Parameter]) -> None:
        """
                            Add new Parameter nodes to the list.
        
                            Method doesn't change or validate graph, it should be done manually.
                            For example, if you want to replace `ReadValue` node by `Parameter`, you should do the
                            following steps:
                            * replace node `ReadValue` by `Parameter` in graph
                            * call add_parameter() to add new input to the list
                            * call graph validation to check correctness of changes
        
                            :param parameter: new Parameter nodes.
                            :type parameter: list[op.Parameter]
        """
    def add_results(self, results: collections.abc.Sequence[op.Result]) -> None:
        """
                            Add new Result nodes to the list.
        
                            Method doesn't validate graph, it should be done manually after all changes.
        
                            :param results: new Result nodes.
                            :type results: list[op.Result]
        """
    def add_sinks(self, sinks: list) -> None:
        """
                    Add new sink nodes to the list.
        
                    Method doesn't validate graph, it should be done manually after all changes.
        
                    :param sinks: new sink nodes.
                    :type sinks: list[openvino.Node]
        """
    def add_variables(self, variables: collections.abc.Sequence[op.util.Variable]) -> None:
        """
                            Add new variables to the list.
        
                            Method doesn't validate graph, it should be done manually after all changes.
        
                            :param variables: new variables to add.
                            :type variables: list[op.util.Variable]
        """
    def clone(self) -> Model:
        """
                    Return a copy of self.
                    :return: A copy of self.
                    :rtype: openvino.Model
        """
    def evaluate(self, output_tensors: collections.abc.Sequence[Tensor], input_tensors: collections.abc.Sequence[Tensor], evaluation_context: RTMap = ...) -> bool:
        """
                    Evaluate the model on inputs, putting results in outputs
        
                    :param output_tensors: Tensors for the outputs to compute. One for each result
                    :type output_tensors: list[openvino.Tensor]
                    :param input_tensors: Tensors for the inputs. One for each inputs.
                    :type input_tensors: list[openvino.Tensor]
                    :param evaluation_context: Storage of additional settings and attributes that can be used
                                               when evaluating the model. This additional information can be
                                               shared across nodes.
                    :type evaluation_context: openvino.RTMap
                    :rtype: bool
        """
    def get_friendly_name(self) -> str:
        """
                            Gets the friendly name for a model. If no
                            friendly name has been set via set_friendly_name
                            then the model's unique name is returned.
        
                            :return: String with a friendly name of the model.
                            :rtype: str
        """
    def get_name(self) -> str:
        """
                            Get the unique name of the model.
        
                            :return: String with a name of the model.
                            :rtype: str
        """
    def get_ops(self) -> list[Node]:
        """
                            Return ops used in the model.
        
                            :return: list of Nodes representing ops used in model.
                            :rtype: list[openvino.Node]
        """
    def get_ordered_ops(self) -> list[Node]:
        """
                            Return ops used in the model in topological order.
        
                            :return: list of sorted Nodes representing ops used in model.
                            :rtype: list[openvino.Node]
        """
    def get_output_element_type(self, index: typing.SupportsInt) -> Type:
        """
                            Return the element type of output i
        
                            :param index: output index
                            :type index: int
                            :return: Type object of output i
                            :rtype: openvino.Type
        """
    def get_output_op(self, index: typing.SupportsInt) -> Node:
        """
                            Return the op that generates output i
        
                            :param index: output index
                            :type index: output index
                            :return: Node object that generates output i
                            :rtype: openvino.Node
        """
    def get_output_partial_shape(self, index: typing.SupportsInt) -> PartialShape:
        """
                            Return the partial shape of element i
        
                            :param index: element index
                            :type index: int
                            :return: PartialShape object of element i
                            :rtype: openvino.PartialShape
        """
    def get_output_shape(self, index: typing.SupportsInt) -> Shape:
        """
                            Return the shape of element i
        
                            :param index: element index
                            :type index: int
                            :return: Shape object of element i
                            :rtype: openvino.Shape
        """
    def get_output_size(self) -> int:
        """
                            Return the number of outputs for the model.
        
                            :return: Number of outputs.
                            :rtype: int
        """
    def get_parameter_index(self, parameter: op.Parameter) -> int:
        """
                            Return the index position of `parameter`
        
                            Return -1 if parameter not matched.
        
                            :param parameter: Parameter, which index is to be found.
                            :type parameter: op.Parameter
                            :return: Index for parameter
                            :rtype: int
        """
    def get_parameters(self) -> list[op.Parameter]:
        """
                            Return the model parameters.
                            
                            :return: a list of model's parameters.
                            :rtype: list[op.Parameter]
        """
    def get_result(self) -> Node:
        """
                            Return single result.
        
                            :return: Node object representing result.
                            :rtype: op.Result
        """
    @typing.overload
    def get_result_index(self, value: Output) -> int:
        """
                            Return index of result.
        
                            Return -1 if `value` not matched.
        
                            :param value: Output containing Node
                            :type value: openvino.Output
                            :return: Index for value referencing it.
                            :rtype: int
        """
    @typing.overload
    def get_result_index(self, value: ConstOutput) -> int:
        """
                            Return index of result.
        
                            Return -1 if `value` not matched.
        
                            :param value: Output containing Node
                            :type value: openvino.Output
                            :return: Index for value referencing it.
                            :rtype: int
        """
    @typing.overload
    def get_result_index(self, result: op.Result) -> int:
        """
                        Return index of result.
        
                        Return -1 if `result` not matched.
        
                        :param result: Result operation
                        :type result: op.Result
                        :return: Index for result referencing it.
                        :rtype: int
        """
    def get_results(self) -> list[op.Result]:
        """
                            Return a list of model outputs.
        
                            :return: a list of model's result nodes.
                            :rtype: list[op.Result]
        """
    @typing.overload
    def get_rt_info(self) -> RTMap:
        """
                        Returns PyRTMap which is a dictionary of user defined runtime info.
        
                        :return: A dictionary of user defined data.
                        :rtype: openvino.RTMap
        """
    @typing.overload
    def get_rt_info(self, path: list) -> typing.Any:
        """
                        Returns runtime attribute as a OVAny object.
        
                        :param path: list of strings which defines a path to runtime info.
                        :type path: list[str]
        
                        :return: A runtime attribute.
                        :rtype: openvino.OVAny
        """
    @typing.overload
    def get_rt_info(self, path: str) -> typing.Any:
        """
                        Returns runtime attribute as a OVAny object.
        
                        :param path: list of strings which defines a path to runtime info.
                        :type path: str
        
                        :return: A runtime attribute.
                        :rtype: openvino.OVAny
        """
    @typing.overload
    def get_sink_index(self, value: Output) -> int:
        """
                            Return index of sink.
        
                            Return -1 if `value` not matched.
        
                            :param value: Output sink node handle
                            :type value: openvino.Output
                            :return: Index of sink node referenced by output handle.
                            :rtype: int
        """
    @typing.overload
    def get_sink_index(self, value: ConstOutput) -> int:
        """
                            Return index of sink.
        
                            Return -1 if `value` not matched.
        
                            :param value: Output sink node handle
                            :type value: openvino.Output
                            :return: Index of sink node referenced by output handle.
                            :rtype: int
        """
    @typing.overload
    def get_sink_index(self, sink: typing.Any) -> int:
        """
                            Return index of sink node.
        
                            Return -1 if `sink` not matched.
        
                            :param sink: Sink node.
                            :type sink: openvino.Node
                            :return: Index of sink node.
                            :rtype: int
        """
    def get_sinks(self) -> list[Node]:
        """
                    Return a list of model's sinks.
        
                    :return: a list of model's sinks.
                    :rtype: list[openvino.Node]
        """
    def get_variable_by_id(self, arg0: str) -> op.util.Variable:
        """
                            Return a variable by specified variable_id.
        
                            :param variable_id: a variable id to get variable node.
                            :type variable_id: str
                            :return: a variable node.
                            :rtype: op.util.Variable
        """
    def get_variables(self) -> list[op.util.Variable]:
        """
                            Return a list of model's variables.
                            
                            :return: a list of model's variables.
                            :rtype: list[op.util.Variable]
        """
    @typing.overload
    def has_rt_info(self, path: list) -> bool:
        """
                        Checks if given path exists in runtime info of the model.
        
                        :param path: list of strings which defines a path to runtime info.
                        :type path: list[str]
        
                        :return: `True` if path exists, otherwise `False`.
                        :rtype: bool
        """
    @typing.overload
    def has_rt_info(self, path: str) -> bool:
        """
                        Checks if given path exists in runtime info of the model.
        
                        :param path: list of strings which defines a path to runtime info.
                        :type path: str
        
                        :return: `True` if path exists, otherwise `False`.
                        :rtype: bool
        """
    @typing.overload
    def input(self) -> Output:
        ...
    @typing.overload
    def input(self, index: typing.SupportsInt) -> Output:
        ...
    @typing.overload
    def input(self, tensor_name: str) -> Output:
        ...
    @typing.overload
    def input(self) -> ConstOutput:
        ...
    @typing.overload
    def input(self, index: typing.SupportsInt) -> ConstOutput:
        ...
    @typing.overload
    def input(self, tensor_name: str) -> ConstOutput:
        ...
    def is_dynamic(self) -> bool:
        """
                            Returns true if any of the op's defined in the model
                            contains partial shape.
        
                            :rtype: bool
        """
    @typing.overload
    def output(self) -> Output:
        ...
    @typing.overload
    def output(self, index: typing.SupportsInt) -> Output:
        ...
    @typing.overload
    def output(self, tensor_name: str) -> Output:
        ...
    @typing.overload
    def output(self) -> ConstOutput:
        ...
    @typing.overload
    def output(self, index: typing.SupportsInt) -> ConstOutput:
        ...
    @typing.overload
    def output(self, tensor_name: str) -> ConstOutput:
        ...
    def remove_parameter(self, parameter: op.Parameter) -> None:
        """
                    Delete Parameter node from the list of parameters. Method will not delete node from graph.
                    You need to replace Parameter with other operation manually.
        
                    Attention: Indexing of parameters can be changed.
        
                    Possible use of method is to replace input by variable. For it the following steps should be done:
                    * `Parameter` node should be replaced by `ReadValue`
                    * call remove_parameter(param) to remove input from the list
                    * check if any parameter indexes are saved/used somewhere, update it for all inputs because indexes can be changed
                    * call graph validation to check all changes
        
                    :param parameter: Parameter node to delete.
                    :type parameter: op.Parameter
        """
    def remove_result(self, result: op.Result) -> None:
        """
                        Delete Result node from the list of results. Method will not delete node from graph.
        
                        :param result: Result node to delete.
                        :type result: op.Result
        """
    def remove_sink(self, sink: typing.Any) -> None:
        """
                        Delete sink node from the list of sinks. Method doesn't delete node from graph.
        
                        :param sink: Sink to delete.
                        :type sink: openvino.Node
        """
    def remove_variable(self, variable: op.util.Variable) -> None:
        """
                            Delete variable from the list of variables.
                            Method doesn't delete nodes that used this variable from the graph.
        
                            :param variable:  Variable to delete.
                            :type variable: op.util.Variable
        """
    def replace_parameter(self, parameter_index: typing.SupportsInt, parameter: op.Parameter) -> None:
        """
                            Replace the `parameter_index` parameter of the model with `parameter`
        
                            All users of the `parameter_index` parameter are redirected to `parameter` , and the
                            `parameter_index` entry in the model parameter list is replaced with `parameter`
        
                            :param parameter_index: The index of the parameter to replace.
                            :type parameter_index: int
                            :param parameter: The parameter to substitute for the `parameter_index` parameter.
                            :type parameter: op.Parameter
        """
    @typing.overload
    def reshape(self, partial_shape: PartialShape, variables_shapes: dict = {}) -> None:
        """
                        Reshape model input.
        
                        The allowed types of keys in the `variables_shapes` dictionary is `str`.
                        The allowed types of values in the `variables_shapes` are:
        
                        (1) `openvino.PartialShape`
                        (2) `list` consisting of dimensions
                        (3) `tuple` consisting of dimensions
                        (4) `str`, string representation of `openvino.PartialShape`
        
                        When list or tuple are used to describe dimensions, each dimension can be written in form:
        
                        (1) non-negative `int` which means static value for the dimension
                        (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                        the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                        max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                        `openvino.Dimension` (5) `str` using next syntax:
                            '?' - to define fully dynamic dimension
                            '1' - to define dimension which length is 1
                            '1..10' - to define bounded dimension
                            '..10' or '1..' to define dimension with only lower or only upper limit
        
                        GIL is released while running this function.
        
                        :param partial_shape: New shape.
                        :type partial_shape: openvino.PartialShape
                        :param variables_shapes: New shapes for variables
                        :type variables_shapes: dict[keys, values]
                        :return : void
        """
    @typing.overload
    def reshape(self, partial_shape: list, variables_shapes: dict = {}) -> None:
        """
                        Reshape model input.
        
                        The allowed types of keys in the `variables_shapes` dictionary is `str`.
                        The allowed types of values in the `variables_shapes` are:
        
                        (1) `openvino.PartialShape`
                        (2) `list` consisting of dimensions
                        (3) `tuple` consisting of dimensions
                        (4) `str`, string representation of `openvino.PartialShape`
        
                        When list or tuple are used to describe dimensions, each dimension can be written in form:
        
                        (1) non-negative `int` which means static value for the dimension
                        (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                        the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                        max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                        `openvino.Dimension` (5) `str` using next syntax:
                            '?' - to define fully dynamic dimension
                            '1' - to define dimension which length is 1
                            '1..10' - to define bounded dimension
                            '..10' or '1..' to define dimension with only lower or only upper limit
        
                        GIL is released while running this function.
        
                        :param partial_shape: New shape.
                        :type partial_shape: list
                        :param variables_shapes: New shapes for variables
                        :type variables_shapes: dict[keys, values]
                        :return : void
        """
    @typing.overload
    def reshape(self, partial_shape: tuple, variables_shapes: dict = {}) -> None:
        """
                        Reshape model input.
        
                        The allowed types of keys in the `variables_shapes` dictionary is `str`.
                        The allowed types of values in the `variables_shapes` are:
        
                        (1) `openvino.PartialShape`
                        (2) `list` consisting of dimensions
                        (3) `tuple` consisting of dimensions
                        (4) `str`, string representation of `openvino.PartialShape`
        
                        When list or tuple are used to describe dimensions, each dimension can be written in form:
        
                        (1) non-negative `int` which means static value for the dimension
                        (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                        the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                        max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                        `openvino.Dimension` (5) `str` using next syntax:
                            '?' - to define fully dynamic dimension
                            '1' - to define dimension which length is 1
                            '1..10' - to define bounded dimension
                            '..10' or '1..' to define dimension with only lower or only upper limit
        
                        GIL is released while running this function.
        
                        :param partial_shape: New shape.
                        :type partial_shape: tuple
                        :param variables_shapes: New shapes for variables
                        :type variables_shapes: dict[keys, values]
                        :return : void
        """
    @typing.overload
    def reshape(self, partial_shape: str, variables_shapes: dict = {}) -> None:
        """
                        Reshape model input.
        
                        The allowed types of keys in the `variables_shapes` dictionary is `str`.
                        The allowed types of values in the `variables_shapes` are:
        
                        (1) `openvino.PartialShape`
                        (2) `list` consisting of dimensions
                        (3) `tuple` consisting of dimensions
                        (4) `str`, string representation of `openvino.PartialShape`
        
                        When list or tuple are used to describe dimensions, each dimension can be written in form:
        
                        (1) non-negative `int` which means static value for the dimension
                        (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                        the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                        max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                        `openvino.Dimension` (5) `str` using next syntax:
                            '?' - to define fully dynamic dimension
                            '1' - to define dimension which length is 1
                            '1..10' - to define bounded dimension
                            '..10' or '1..' to define dimension with only lower or only upper limit
        
        
                        GIL is released while running this function.
        
                        :param partial_shape: New shape.
                        :type partial_shape: str
                        :param variables_shapes: New shapes for variables
                        :type variables_shapes: dict[keys, values]
                        :return : void
        """
    @typing.overload
    def reshape(self, partial_shapes: dict, variables_shapes: dict = {}) -> None:
        """
         Reshape model inputs.
        
                    The allowed types of keys in the `partial_shapes` dictionary are:
        
                    (1) `int`, input index
                    (2) `str`, input tensor name
                    (3) `openvino.Output`
        
                    The allowed types of values in the `partial_shapes` are:
        
                    (1) `openvino.PartialShape`
                    (2) `list` consisting of dimensions
                    (3) `tuple` consisting of dimensions
                    (4) `str`, string representation of `openvino.PartialShape`
        
                    When list or tuple are used to describe dimensions, each dimension can be written in form:
        
                    (1) non-negative `int` which means static value for the dimension
                    (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound; the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound
                    (3) `(min, max)`, the same as above
                    (4) `-1` is a dynamic dimension without known bounds
                    (4) `openvino.Dimension`
                    (5) `str` using next syntax:
                        '?' - to define fully dynamic dimension
                        '1' - to define dimension which length is 1
                        '1..10' - to define bounded dimension
                        '..10' or '1..' to define dimension with only lower or only upper limit
        
                    The allowed types of keys in the `variables_shapes` dictionary is `str`.
                    The allowed types of values in the `variables_shapes` are:
        
                    (1) `openvino.PartialShape`
                    (2) `list` consisting of dimensions
                    (3) `tuple` consisting of dimensions
                    (4) `str`, string representation of `openvino.PartialShape`
        
                    When list or tuple are used to describe dimensions, each dimension can be written in form:
        
                    (1) non-negative `int` which means static value for the dimension
                    (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                    the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                    max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                    `openvino.Dimension` (5) `str` using next syntax:
                        '?' - to define fully dynamic dimension
                        '1' - to define dimension which length is 1
                        '1..10' - to define bounded dimension
                        '..10' or '1..' to define dimension with only lower or only upper limit
        
                    Reshape model inputs.
        
                    GIL is released while running this function.
        
                    :param partial_shapes: New shapes.
                    :type partial_shapes: dict[keys, values]
                    :param variables_shapes: New shapes for variables
                    :type variables_shapes: dict[keys, values]
        """
    def set_friendly_name(self, name: str) -> None:
        """
                            Sets a friendly name for a model. This does
                            not overwrite the unique name of the model and
                            is retrieved via get_friendly_name(). Used mainly
                            for debugging.
        
                            :param name: String to set as the friendly name.
                            :type name: str
        """
    @typing.overload
    def set_rt_info(self, obj: typing.Any, path: list) -> None:
        """
                        Add value inside runtime info
        
                        :param obj: value for the runtime info
                        :type obj: py:object
                        :param path: list of strings which defines a path to runtime info.
                        :type path: list[str]
        """
    @typing.overload
    def set_rt_info(self, obj: typing.Any, path: str) -> None:
        """
                        Add value inside runtime info
        
                        :param obj: value for the runtime info
                        :type obj: Any
                        :param path: String which defines a path to runtime info.
                        :type path: str
        """
    def validate_nodes_and_infer_types(self) -> None:
        ...
    @property
    def dynamic(self) -> bool:
        """
                                                Returns true if any of the op's defined in the model
                                                contains partial shape.
        
                                                :rtype: bool
        """
    @property
    def inputs(self) -> list[Output]:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def outputs(self) -> list[Output]:
        ...
    @property
    def parameters(self) -> list[op.Parameter]:
        """
                                                Return the model parameters.
                                                
                                                :return: a list of model's parameters.
                                                :rtype: list[op.Parameter]
        """
    @property
    def result(self) -> Node:
        """
                                                Return single result.
        
                                                :return: Node object representing result.
                                                :rtype: op.Result
        """
    @property
    def results(self) -> list[op.Result]:
        """
                                            Return a list of model outputs.
        
                                            :return: a list of model's result nodes.
                                            :rtype: list[op.Result]
        """
    @property
    def rt_info(self) -> RTMap:
        ...
    @property
    def sinks(self) -> list[Node]:
        """
                    Return a list of model's sinks.
        
                    :return: a list of model's sinks.
                    :rtype: list[openvino.Node]
        """
    @property
    def variables(self) -> list[op.util.Variable]:
        """
                                            Return a list of model's variables.
                                            
                                            :return: a list of model's variables.
                                            :rtype: list[op.util.Variable]
        """
class Node:
    """
    openvino.Node wraps ov::Node
    """
    friendly_name: str
    def __add__(self, right: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        """
                    Return node which applies f(A,B) = A+B to the input nodes element-wise.
        
                    :param right: The right operand.
                    :type right: Union[openvino.Node, int, float, numpy.ndarray]
                    :return: The node performing element-wise addition.
                    :rtype: openvino.Node
        """
    def __array_ufunc__(self, arg0: typing.Any, arg1: str, *args, **kwargs) -> typing.Any:
        ...
    def __getattr__(self, arg0: str) -> collections.abc.Callable:
        ...
    def __mul__(self, right: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        """
                    Return node which applies f(A,B) = A*B to the input nodes element-wise.
        
                    :param right: The right operand.
                    :type right: Union[openvino.Node, int, float, numpy.ndarray]
                    :return: The node performing element-wise multiplication.
                    :rtype: openvino.Node
        """
    def __radd__(self, arg0: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        ...
    def __repr__(self) -> str:
        ...
    def __rmul__(self, arg0: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        ...
    def __rsub__(self, arg0: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        ...
    def __rtruediv__(self, arg0: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        ...
    def __sub__(self, right: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        """
                    Return node which applies f(A,B) = A-B to the input nodes element-wise.
        
                    :param right: The right operand.
                    :type right: Union[openvino.Node, int, float, numpy.ndarray]
                    :return: The node performing element-wise subtraction.
                    :rtype: openvino.Node
        """
    def __truediv__(self, right: openvino._pyopenvino.Node | typing.SupportsInt | typing.SupportsFloat | numpy.ndarray) -> Node:
        """
                    Return node which applies f(A,B) = A/B to the input nodes element-wise.
        
                    :param right: The right operand.
                    :type right: Union[openvino.Node, int, float, numpy.ndarray]
                    :return: The node performing element-wise division.
                    :rtype: openvino.Node
        """
    def constructor_validate_and_infer_types(self) -> None:
        ...
    @typing.overload
    def evaluate(self, output_values: collections.abc.Sequence[Tensor], input_values: collections.abc.Sequence[Tensor], evaluationContext: RTMap) -> bool:
        """
                        Evaluate the node on inputs, putting results in outputs
                        
                        :param output_tensors: Tensors for the outputs to compute. One for each result.
                        :type output_tensors: list[openvino.Tensor]
                        :param input_tensors: Tensors for the inputs. One for each inputs.
                        :type input_tensors: list[openvino.Tensor]
                        :param evaluation_context: Storage of additional settings and attributes that can be used
                        when evaluating the function. This additional information can be shared across nodes.
                        :type evaluation_context: openvino.RTMap
                        :rtype: bool
        """
    @typing.overload
    def evaluate(self, output_values: collections.abc.Sequence[Tensor], input_values: collections.abc.Sequence[Tensor]) -> bool:
        """
                        Evaluate the function on inputs, putting results in outputs
        
                        :param output_tensors: Tensors for the outputs to compute. One for each result.
                        :type output_tensors: list[openvino.Tensor]
                        :param input_tensors: Tensors for the inputs. One for each inputs.
                        :type input_tensors: list[openvino.Tensor]
                        :rtype: bool
        """
    def get_attributes(self) -> dict:
        ...
    def get_element_type(self) -> Type:
        """
                        Checks that there is exactly one output and returns it's element type.
        
                        :return: Type of the output.
                        :rtype: openvino.Type
        """
    def get_friendly_name(self) -> str:
        """
                        Gets the friendly name for a node. If no friendly name has
                        been set via set_friendly_name then the node's unique name
                        is returned.
        
                        :return: Friendly name of the node.
                        :rtype: str
        """
    def get_input_element_type(self, index: typing.SupportsInt) -> Type:
        """
                        Returns the element type for input index
        
                        :param index: Index of the input.
                        :type index: int
                        :return: Type of the input index
                        :rtype: openvino.Type
        """
    def get_input_partial_shape(self, index: typing.SupportsInt) -> PartialShape:
        """
                        Returns the partial shape for input index
        
                        :param index: Index of the input.
                        :type index: int
                        :return: PartialShape of the input index
                        :rtype: openvino.PartialShape
        """
    def get_input_shape(self, index: typing.SupportsInt) -> Shape:
        """
                        Returns the shape for input index
        
                        :param index: Index of the input.
                        :type index: int
                        :return: Shape of the input index
                        :rtype: openvino.Shape
        """
    def get_input_size(self) -> int:
        """
                        Returns the number of inputs to the node.
        
                        :return: Number of inputs.
                        :rtype: int
        """
    def get_input_tensor(self, index: typing.SupportsInt) -> DescriptorTensor:
        """
                        Returns the tensor for the node's input with index i
        
                        :param index: Index of Input.
                        :type index: int
                        :return: Tensor of the input index
                        :rtype: openvino._pyopenvino.DescriptorTensor
        """
    def get_instance_id(self) -> int:
        """
                        Returns id of the node.
                        May be used to compare nodes if they are same instances.
        
                        :return: id of the node.
                        :rtype: int
        """
    def get_name(self) -> str:
        """
                        Get the unique name of the node
        
                        :return: Unique name of the node.
                        :rtype: str
        """
    def get_output_element_type(self, index: typing.SupportsInt) -> Type:
        """
                        Returns the element type for output index
        
                        :param index: Index of the output.
                        :type index: int
                        :return: Type of the output index
                        :rtype: openvino.Type
        """
    def get_output_partial_shape(self, index: typing.SupportsInt) -> PartialShape:
        """
                        Returns the partial shape for output index
        
                        :param index: Index of the output.
                        :type index: int
                        :return: PartialShape of the output index
                        :rtype: openvino.PartialShape
        """
    def get_output_shape(self, index: typing.SupportsInt) -> Shape:
        """
                        Returns the shape for output index
        
                        :param index: Index of the output.
                        :type index: int
                        :return: Shape of the output index
                        :rtype: openvino.Shape
        """
    def get_output_size(self) -> int:
        """
                        Returns the number of outputs from the node.
        
                        :return: Number of outputs.
                        :rtype: int
        """
    def get_output_tensor(self, index: typing.SupportsInt) -> DescriptorTensor:
        """
                        Returns the tensor for output index
        
                        :param index: Index of the output.
                        :type index: int
                        :return: Tensor of the output index
                        :rtype: openvino._pyopenvino.DescriptorTensor
        """
    def get_rt_info(self) -> RTMap:
        """
                        Returns RTMap which is a dictionary of user defined runtime info.
        
                        :return: A dictionary of user defined data.
                        :rtype: openvino.RTMap
        """
    def get_type_info(self) -> DiscreteTypeInfo:
        ...
    def get_type_name(self) -> str:
        """
                        Returns Type's name from the node.
        
                        :return: String representing Type's name.
                        :rtype: str
        """
    def input(self, input_index: typing.SupportsInt) -> Input:
        """
                        A handle to the input_index input of this node.
        
                        :param input_index: Index of Input.
                        :type input_index: int
                        :return: Input of this node.
                        :rtype: openvino.Input
        """
    def input_value(self, index: typing.SupportsInt) -> Output:
        """
                        Returns input of the node with index i
        
                        :param index: Index of Input.
                        :type index: int
                        :return: Input of this node.
                        :rtype: openvino.Input
        """
    def input_values(self) -> list[Output]:
        """
                         Returns list of node's inputs, in order.
        
                         :return: list of node's inputs
                         :rtype: list[openvino.Input]
        """
    def inputs(self) -> list[Input]:
        """
                        A list containing a handle for each of this node's inputs, in order.
        
                        :return: list of node's inputs.
                        :rtype: list[openvino.Input]
        """
    def output(self, output_index: typing.SupportsInt) -> Output:
        """
                        A handle to the output_index output of this node.
        
                        :param output_index: Index of Output.
                        :type output_index: int
                        :return: Output of this node.
                        :rtype: openvino.Output
        """
    def outputs(self) -> list[Output]:
        """
                        A list containing a handle for each of this node's outputs, in order.
        
                        :return: list of node's outputs.
                        :rtype: list[openvino.Output]
        """
    def set_argument(self, arg0: typing.SupportsInt, arg1: Output) -> None:
        ...
    @typing.overload
    def set_arguments(self, arg0: collections.abc.Sequence[Node]) -> None:
        ...
    @typing.overload
    def set_arguments(self, arg0: collections.abc.Sequence[Output]) -> None:
        ...
    def set_attribute(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def set_friendly_name(self, name: str) -> None:
        """
                        Sets a friendly name for a node. This does not overwrite the unique name
                        of the node and is retrieved via get_friendly_name(). Used mainly for
                        debugging. The friendly name may be set exactly once.
        
                        :param name: Friendly name to set.
                        :type name: str
        """
    def set_output_size(self, size: typing.SupportsInt) -> None:
        """
                        Sets the number of outputs
        
                        :param size: number of outputs.
                        :type size: int
        """
    def set_output_type(self, index: typing.SupportsInt, element_type: Type, shape: PartialShape) -> None:
        """
                        Sets output's element type and shape.
        
                        :param index: Index of the output.
                        :type index: int
                        :param element_type: Element type of the output.
                        :type element_type: openvino.Type
                        :param shape: Shape of the output.
                        :type shape: openvino.PartialShape
        """
    def set_rt_info(self, value: typing.Any, key: str) -> None:
        """
                        Add a value to the runtime info.
        
                        :param value: Value for the runtime info.
                        :type value: Any
                        :param key: String that defines a key in the runtime info dictionary.
                        :type key: str
        """
    def validate_and_infer_types(self) -> None:
        """
                Verifies that attributes and inputs are consistent and computes output shapes and element types.
                Must be implemented by concrete child classes so that it can be run any number of times.
                
                Throws if the node is invalid.
        """
    def visit_attributes(self, arg0: AttributeVisitor) -> bool:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def rt_info(self) -> RTMap:
        ...
    @property
    def shape(self) -> Shape:
        ...
    @property
    def type_info(self) -> DiscreteTypeInfo:
        ...
class NodeContext:
    def get_attribute(self, name: str, default_value: typing.Any = None, dtype: typing.Any = None) -> typing.Any:
        ...
    @typing.overload
    def get_input(self, arg0: typing.SupportsInt) -> Output:
        ...
    @typing.overload
    def get_input(self, arg0: str) -> Output:
        ...
    @typing.overload
    def get_input(self, arg0: str, arg1: typing.SupportsInt) -> Output:
        ...
    @typing.overload
    def get_input_size(self) -> int:
        ...
    @typing.overload
    def get_input_size(self, arg0: str) -> int:
        ...
    def get_op_type(self, arg0: str) -> str:
        ...
    def get_values_from_const_input(self, idx: typing.SupportsInt, default_value: typing.Any = None, dtype: typing.Any = None) -> typing.Any:
        ...
    def has_attribute(self, arg0: str) -> bool:
        ...
class NodeFactory:
    """
    NodeFactory creates nGraph nodes
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def add_extension(self, arg0: Extension) -> None:
        ...
    @typing.overload
    def add_extension(self, arg0: collections.abc.Sequence[Extension]) -> None:
        ...
    @typing.overload
    def add_extension(self, arg0: typing.Any) -> None:
        ...
    @typing.overload
    def create(self, arg0: str) -> Node:
        ...
    @typing.overload
    def create(self, arg0: str, arg1: collections.abc.Sequence[Output], arg2: dict) -> Node:
        ...
class NotImplementedFailure(Exception):
    pass
class OVAny:
    """
    openvino.OVAny provides object wrapper for OpenVINOov::Any class. It allows to pass different types of objectsinto C++ based core of the project.
    """
    @typing.overload
    def __eq__(self, arg0: OVAny) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: typing.Any) -> bool:
        ...
    def __get__(self) -> typing.Any:
        ...
    def __getitem__(self, arg0: typing.Any) -> typing.Any:
        ...
    def __hash__(self) -> typing.Any:
        ...
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def __len__(self) -> typing.Any:
        ...
    def __repr__(self) -> str:
        ...
    def __set__(self, arg0: OVAny) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.Any, arg1: str) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.Any, arg1: typing.SupportsInt) -> None:
        ...
    def aslist(self, dtype: typing.Any = None) -> typing.Any:
        """
                    Returns runtime attribute as a list with specified data type.
        
                    :param dtype: Data type of a list in which runtime attribute will be casted.
                    :type dtype: Union[bool, int, str, float]
        
                    :return: A runtime attribute as a list.
                    :rtype: Union[list[float], list[int], list[str], list[bool]]
        """
    def astype(self, arg0: typing.Any) -> typing.Any:
        """
                    Returns runtime attribute casted to defined data type.
        
                    :param dtype: Data type in which runtime attribute will be casted.
                    :type dtype: Union[bool, int, str, float, dict]
        
                    :return: A runtime attribute.
                    :rtype: Any
        """
    def get(self) -> typing.Any:
        """
                    :return: Value of this OVAny.
                    :rtype: Any
        """
    def set(self, arg0: typing.Any) -> None:
        """
                    :param: Value to be set in OVAny.
                    :type: Any
        """
    @property
    def value(self) -> typing.Any:
        """
                    :return: Value of this OVAny.
                    :rtype: Any
        """
class Op(Node):
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def _update_type_info(self) -> None:
        ...
class OpConversionFailure(Exception):
    pass
class OpExtension(Extension):
    """
    openvino.OpExtension provides the base interface for OpenVINO extensions.
    """
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def __repr__(self) -> str:
        ...
class OpValidationFailure(Exception):
    pass
class Output:
    """
    openvino.Output represents port/node output.
    """
    def __copy__(self) -> Output:
        ...
    def __deepcopy__(self, arg0: dict) -> None:
        ...
    def __eq__(self, arg0: Output) -> bool:
        ...
    def __ge__(self, arg0: Output) -> bool:
        ...
    def __gt__(self, arg0: Output) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __le__(self, arg0: Output) -> bool:
        ...
    def __lt__(self, arg0: Output) -> bool:
        ...
    def __ne__(self, arg0: Output) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def _from_node(self: typing.Any) -> Output:
        ...
    def add_names(self, names: collections.abc.Set[str]) -> None:
        """
                    Add tensor names associated with this output.
        
                    :param names: set of tensor names.
                    :type names: set[str]
        """
    def get_any_name(self) -> str:
        """
                        One of the tensor names associated with this output.
                        Note: first name in lexicographical order.
        
                        :return: Tensor name as string.
                        :rtype: str
        """
    def get_element_type(self) -> Type:
        """
                        The element type of the output referred to by this output handle.
        
                        :return: Type of the output.
                        :rtype: openvino.Type
        """
    def get_index(self) -> int:
        """
                        The index of the output referred to by this output handle.
        
                        :return: Index value as integer.
                        :rtype: int
        """
    def get_names(self) -> set[str]:
        """
                        The tensor names associated with this output.
        
                        :return: set of tensor names.
                        :rtype: set[str]
        """
    def get_node(self) -> ...:
        """
                        Get node referenced by this output handle.
        
                        :return: Node object referenced by this output handle.
                        :rtype: openvino.Node
        """
    def get_partial_shape(self) -> PartialShape:
        """
                        The partial shape of the output referred to by this output handle.
        
                        :return: Copy of PartialShape of the output.
                        :rtype: openvino.PartialShape
        """
    def get_rt_info(self) -> RTMap:
        """
                        Returns RTMap which is a dictionary of user defined runtime info.
        
                        :return: A dictionary of user defined data.
                        :rtype: openvino.RTMap
        """
    def get_shape(self) -> Shape:
        """
                        The shape of the output referred to by this output handle.
        
                        :return: Copy of Shape of the output.
                        :rtype: openvino.Shape
        """
    def get_target_inputs(self) -> set[...]:
        """
                        A set containing handles for all inputs, targeted by the output,
                        referenced by this output handle.
        
                        :return: set of Inputs.
                        :rtype: set[openvino.Input]
        """
    def get_tensor(self) -> ...:
        """
                        A reference to the tensor descriptor for this output.
        
                        :return: Tensor of the output.
                        :rtype: openvino._pyopenvino.DescriptorTensor
        """
    def remove_target_input(self, target_input: typing.Any) -> None:
        """
                        Removes a target input from the output referenced by this output handle.
        
                        :param target_input: The target input to remove.
                        :type target_input: openvino.Output
        """
    def replace(self, replacement: Output) -> None:
        """
                        Replace all users of this value with replacement.
        
                        :param replacement: The node that is a replacement.
                        :type replacement: openvino.Output
        """
    def set_names(self, names: collections.abc.Set[str]) -> None:
        """
                    Set tensor names associated with this output.
        
                    :param names: set of tensor names.
                    :type names: set[str]
        """
    def set_rt_info(self, value: typing.Any, key: str) -> None:
        """
                        Add a value to the runtime info.
        
                        :param value: Value for the runtime info.
                        :type value: Any
                        :param key: String that defines a key in the runtime info dictionary.
                        :type key: str
        """
    @property
    def any_name(self) -> str:
        ...
    @property
    def element_type(self) -> Type:
        ...
    @property
    def index(self) -> int:
        ...
    @property
    def names(self) -> set[str]:
        ...
    @property
    def node(self) -> ...:
        ...
    @property
    def partial_shape(self) -> PartialShape:
        ...
    @property
    def rt_info(self) -> RTMap:
        ...
    @property
    def shape(self) -> Shape:
        ...
    @property
    def target_inputs(self) -> set[...]:
        ...
    @property
    def tensor(self) -> ...:
        ...
class PartialShape:
    """
    openvino.PartialShape wraps ov::PartialShape
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    @typing.overload
    def dynamic(rank: Dimension = ...) -> PartialShape:
        """
                               Construct a PartialShape with the given rank and all dimensions are dynamic.
        
                               :param rank: The rank of the PartialShape. This is the number of dimensions in the shape.
                               :type rank: openvino.Dimension
                               :return: A PartialShape with the given rank (or undefined rank if not provided), and all dimensions are dynamic.
        """
    @staticmethod
    @typing.overload
    def dynamic(rank: typing.SupportsInt) -> PartialShape:
        """
                    Construct a PartialShape with the given rank and all dimensions are dynamic.
        
                    :param rank: The rank of the PartialShape. This is the number of dimensions in the shape.
                    :type rank: int
                    :return: A PartialShape with the given rank, and all dimensions are dynamic.
        """
    def __copy__(self) -> PartialShape:
        ...
    def __deepcopy__(self, arg0: dict) -> PartialShape:
        """
        memo
        """
    @typing.overload
    def __eq__(self, arg0: PartialShape) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: Shape) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: tuple) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: list) -> bool:
        ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> Dimension:
        ...
    @typing.overload
    def __getitem__(self, arg0: slice) -> PartialShape:
        ...
    @typing.overload
    def __init__(self, arg0: Shape) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: PartialShape) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: tuple) -> None:
        ...
    @typing.overload
    def __init__(self, shape: str) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[Dimension]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Dimension) -> None:
        ...
    def __str__(self) -> str:
        ...
    def compatible(self, shape: PartialShape) -> bool:
        """
                        Check whether this shape is compatible with the argument, i.e.,
                        whether it is possible to merge them.
        
                        :param shape: The shape to be checked for compatibility with this shape.
                        :type shape: openvino.PartialShape
                        :return: True if this shape is compatible with s, else False.
                        :rtype: bool
        """
    def get_dimension(self, index: typing.SupportsInt) -> Dimension:
        """
                    Get the dimension at specified index of a partial shape.
        
                    :param index: The index of dimension.
                    :type index: int 
                    :return: Get the particular dimension of a partial shape.
                    :rtype: openvino.Dimension
        """
    def get_max_shape(self) -> Shape:
        """
                        :return: Get the max bounding shape.
                        :rtype: openvino.Shape
        """
    def get_min_shape(self) -> Shape:
        """
                        :return: Get the min bounding shape.
                        :rtype: openvino.Shape
        """
    def get_shape(self) -> Shape:
        """
                        :return: Get the unique shape.
                        :rtype: openvino.Shape
        """
    def refines(self, shape: PartialShape) -> bool:
        """
                        Check whether this shape is a refinement of the argument.
        
                        :param shape: The shape which is being compared against this shape.
                        :type shape: openvino.PartialShape
                        :return: True if this shape refines s, else False.
                        :rtype: bool
        """
    def relaxes(self, shape: PartialShape) -> bool:
        """
                        Check whether this shape is a relaxation of the argument.
        
                        :param shape: The shape which is being compared against this shape.
                        :type shape: openvino.PartialShape
                        :return: True if this shape relaxes s, else False.
                        :rtype: bool
        """
    def same_scheme(self, shape: PartialShape) -> bool:
        """
                        Check whether this shape represents the same scheme as the argument.
        
                        :param shape: The shape which is being compared against this shape.
                        :type shape: openvino.PartialShape
                        :return: True if shape represents the same scheme as s, else False.
                        :rtype: bool
        """
    def to_shape(self) -> Shape:
        """
                        :return: Get the unique shape.
                        :rtype: openvino.Shape
        """
    def to_string(self) -> str:
        ...
    @property
    def all_non_negative(self) -> bool:
        """
                                            True if all static dimensions of the tensor are
                                            non-negative, else False.
        """
    @property
    def is_dynamic(self) -> bool:
        """
                                            False if this shape is static, else True.
                                            A shape is considered static if it has static rank,
                                            and all dimensions of the shape are static.
        """
    @property
    def is_static(self) -> bool:
        """
                                            True if this shape is static, else False.
                                            A shape is considered static if it has static rank,
                                            and all dimensions of the shape are static.
        """
    @property
    def rank(self) -> Dimension:
        """
                                            The rank of the shape.
        """
class Place:
    """
    openvino.frontend.Place wraps ov::frontend::Place
    """
    def get_consuming_operations(self, output_name: typing.Any = None, output_port_index: typing.Any = None) -> list[Place]:
        """
                        Returns references to all operation nodes that consume data from this place for specified output port.
                        Note: It can be called for any kind of graph place searching for the first consuming operations.
        
                        :param output_name: Name of output port group. May not be set if node has one output port group.
                        :type output_name: str
                        :param output_port_index: If place is an operational node it specifies which output port should be considered
                            May not be set if node has only one output port.
                        :type output_port_index: int
                        :return: A list with all operation node references that consumes data from this place
                        :rtype: list[openvino.frontend.Place]
        """
    def get_consuming_ports(self) -> list[Place]:
        """
                        Returns all input ports that consume data flows through this place.
        
                        :return: Input ports that consume data flows through this place.
                        :rtype: list[openvino.frontend.Place]
        """
    def get_input_port(self, input_name: typing.Any = None, input_port_index: typing.Any = None) -> Place:
        """
                        For operation node returns reference to an input port with specified name and index.
        
                        :param input_name: Name of port group. May not be set if node has one input port group.
                        :type input_name: str
                        :param input_port_index: Input port index in a group. May not be set if node has one input port in a group.
                        :type input_port_index: int
                        :return: Appropriate input port place.
                        :rtype: openvino.frontend.Place
        """
    def get_names(self) -> list[str]:
        """
                        All associated names (synonyms) that identify this place in the graph in a framework specific way.
        
                        :return: A vector of strings each representing a name that identifies this place in the graph.
                                 Can be empty if there are no names associated with this place or name cannot be attached.
                        :rtype: list[str]
        """
    def get_output_port(self, output_name: typing.Any = None, output_port_index: typing.Any = None) -> Place:
        """
                        For operation node returns reference to an output port with specified name and index.
        
                        :param output_name: Name of output port group. May not be set if node has one output port group.
                        :type output_name: str
                        :param output_port_index: Output port index. May not be set if node has one output port in a group.
                        :type output_port_index: int
                        :return: Appropriate output port place.
                        :rtype: openvino.frontend.Place
        """
    def get_producing_operation(self, input_name: typing.Any = None, input_port_index: typing.Any = None) -> Place:
        """
                        Get an operation node place that immediately produces data for this place.
        
                        :param input_name: Name of port group. May not be set if node has one input port group.
                        :type input_name: str
                        :param input_port_index: If a given place is itself an operation node, this specifies a port index.
                            May not be set if place has only one input port.
                        :type input_port_index: int
                        :return: An operation place that produces data for this place.
                        :rtype: openvino.frontend.Place
        """
    def get_producing_port(self) -> Place:
        """
                        Returns a port that produces data for this place.
        
                        :return: A port place that produces data for this place.
                        :rtype: openvino.frontend.Place
        """
    def get_source_tensor(self, input_name: typing.Any = None, input_port_index: typing.Any = None) -> Place:
        """
                        Returns a tensor place that supplies data for this place; applicable for operations,
                        input ports and input edges.
        
                        :param input_name : Name of port group. May not be set if node has one input port group.
                        :type input_name: str
                        :param input_port_index: Input port index for operational node. May not be specified if place has only one input port.
                        :type input_port_index: int
                        :return: A tensor place which supplies data for this place.
                        :rtype: openvino.frontend.Place
        """
    def get_target_tensor(self, output_name: typing.Any = None, output_port_index: typing.Any = None) -> Place:
        """
                        Returns a tensor place that gets data from this place; applicable for operations,
                        output ports and output edges.
        
                        :param output_name: Name of output port group. May not be set if node has one output port group.
                        :type output_name: str
                        :param output_port_index: Output port index if the current place is an operation node and has multiple output ports.
                            May not be set if place has only one output port.
                        :type output_port_index: int
                        :return: A tensor place which hold the resulting value for this place.
                        :rtype: openvino.frontend.Place
        """
    def is_equal(self, other: Place) -> bool:
        """
                        Returns true if another place is the same as this place.
        
                        :param other: Another place object.
                        :type other: openvino.frontend.Place
                        :return: True if another place is the same as this place.
                        :rtype: bool
        """
    def is_equal_data(self, other: Place) -> bool:
        """
                        Returns true if another place points to the same data.
                        Note: The same data means all places on path:
                              output port -> output edge -> tensor -> input edge -> input port.
        
                        :param other: Another place object.
                        :type other: openvino.frontend.Place
                        :return: True if another place points to the same data.
                        :rtype: bool
        """
    def is_input(self) -> bool:
        """
                        Returns true if this place is input for a model.
        
                        :return: True if this place is input for a model
                        :rtype: bool
        """
    def is_output(self) -> bool:
        """
                        Returns true if this place is output for a model.
        
                        :return: True if this place is output for a model.
                        :rtype: bool
        """
class ProfilingInfo:
    """
    openvino.ProfilingInfo contains performance metrics for single node.
    """
    class Status:
        """
        Members:
        
          NOT_RUN
        
          OPTIMIZED_OUT
        
          EXECUTED
        """
        EXECUTED: typing.ClassVar[ProfilingInfo.Status]  # value = <Status.EXECUTED: 2>
        NOT_RUN: typing.ClassVar[ProfilingInfo.Status]  # value = <Status.NOT_RUN: 0>
        OPTIMIZED_OUT: typing.ClassVar[ProfilingInfo.Status]  # value = <Status.OPTIMIZED_OUT: 1>
        __members__: typing.ClassVar[dict[str, ProfilingInfo.Status]]  # value = {'NOT_RUN': <Status.NOT_RUN: 0>, 'OPTIMIZED_OUT': <Status.OPTIMIZED_OUT: 1>, 'EXECUTED': <Status.EXECUTED: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    EXECUTED: typing.ClassVar[ProfilingInfo.Status]  # value = <Status.EXECUTED: 2>
    NOT_RUN: typing.ClassVar[ProfilingInfo.Status]  # value = <Status.NOT_RUN: 0>
    OPTIMIZED_OUT: typing.ClassVar[ProfilingInfo.Status]  # value = <Status.OPTIMIZED_OUT: 1>
    cpu_time: datetime.timedelta
    exec_type: str
    node_name: str
    node_type: str
    real_time: datetime.timedelta
    status: ProfilingInfo.Status
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class ProgressReporterExtension(Extension):
    """
    An extension class intented to use as progress reporting utility
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Callable) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Callable[[typing.SupportsFloat, typing.SupportsInt, typing.SupportsInt], None]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Callable[[typing.SupportsFloat, typing.SupportsInt, typing.SupportsInt], None]) -> None:
        ...
    def report_progress(self, arg0: typing.SupportsFloat, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> None:
        ...
class RTMap:
    """
    openvino.RTMap makes bindings for std::map<std::string, ov::Any>, which can later be used as ov::Node::RTMap
    """
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    def __contains__(self, arg0: str) -> bool:
        ...
    def __delitem__(self, arg0: str) -> None:
        ...
    def __getitem__(self, arg0: str) -> typing.Any:
        ...
    def __iter__(self) -> collections.abc.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: str, arg1: str) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: str, arg1: typing.SupportsInt) -> None:
        ...
    def items(self) -> typing.Iterator:
        ...
    def keys(self) -> collections.abc.Iterator[str]:
        ...
    def values(self) -> typing.Iterator:
        ...
class RemoteContext:
    def create_host_tensor(self, type: Type, shape: Shape) -> Tensor:
        """
                    This method is used to create a host tensor object friendly for the device in
                    current context. For example, GPU context may allocate USM host memory
                    (if corresponding extension is available), which could be more efficient
                    than regular host memory.
        
                    GIL is released while running this function.
        
                    :param type: Defines the element type of the tensor.
                    :type type: openvino.Type
                    :param shape: Defines the shape of the tensor.
                    :type shape: openvino.Shape
                    :return: A tensor instance with device friendly memory.
                    :rtype: openvino.Tensor
        """
    def create_tensor(self, type: Type, shape: Shape, properties: collections.abc.Mapping[str, typing.Any]) -> RemoteTensor:
        """
                    Allocates memory tensor in device memory or wraps user-supplied memory handle
                    using the specified tensor description and low-level device-specific parameters.
                    Returns the object that implements the RemoteTensor interface.
        
                    GIL is released while running this function.
        
                    :param type: Defines the element type of the tensor.
                    :type type: openvino.Type
                    :param shape: Defines the shape of the tensor.
                    :type shape: openvino.Shape
                    :param properties: dict of the low-level tensor object parameters.
                    :type properties: dict
                    :return: A remote tensor instance.
                    :rtype: openvino.RemoteTensor
        """
    def get_device_name(self) -> str:
        """
                Returns name of a device on which the context is allocated.
        
                :return: A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]`.
                :rtype: str
        """
    def get_params(self) -> dict[str, OVAny]:
        """
                Returns a dict of device-specific parameters required for low-level
                operations with the underlying context.
                Parameters include device/context handles, access flags, etc.
                Content of the returned dict depends on remote execution context that is
                currently set on the device (working scenario).
        
                :return: A dictionary of device-specific parameters.
                :rtype: dict
        """
class RemoteTensor:
    def __init__(self, remote_tensor: RemoteTensor, begin: Coordinate, end: Coordinate) -> None:
        """
                Constructs a RoiRemoteTensor object using a specified range of coordinates on an existing RemoteTensor.
        
                :param remote_tensor: The RemoteTensor object on which the RoiRemoteTensor will be based.
                :type remote_tensor: openvino.RemoteTensor
                :param begin: The starting coordinates for the tensor bound.
                :type begin: openvino.Coordinate
                :param end: The ending coordinates for the tensor bound.
                :type end: openvino.Coordinate
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def copy_from(self, source_tensor: RemoteTensor) -> None:
        """
                Copy source remote tensor's data to this tensor. Tensors should have the same element type.
                In case of RoiTensor, tensors should also have the same shape.
        
                :param source_tensor: The source remote tensor from which the data will be copied.
                :type source_tensor: openvino.RemoteTensor
        """
    @typing.overload
    def copy_from(self, source_tensor: Tensor) -> None:
        """
                Copy source tensor's data to this tensor. Tensors should have the same element type and shape.
                In case of RoiTensor, tensors should also have the same shape.
        
                :param source_tensor: The source tensor from which the data will be copied.
                :type source_tensor: openvino.Tensor
        """
    @typing.overload
    def copy_to(self, target_tensor: RemoteTensor) -> None:
        """
                Copy tensor's data to a destination remote tensor. The destination tensor should have the same element type.
                In case of RoiTensor, the destination tensor should also have the same shape.
        
                :param target_tensor: The destination remote tensor to which the data will be copied.
                :type target_tensor: openvino.RemoteTensor
        """
    @typing.overload
    def copy_to(self, target_tensor: Tensor) -> None:
        """
                Copy tensor's data to a destination tensor. The destination tensor should have the same element type.
                In case of RoiTensor, the destination tensor should also have the same shape.
        
                :param target_tensor: The destination tensor to which the data will be copied.
                :type target_tensor: openvino.Tensor
        """
    def get_byte_size(self) -> int:
        """
                Gets Tensor's size in bytes.
        
                :rtype: int
        """
    def get_device_name(self) -> str:
        """
                Returns name of a device on which the tensor is allocated.
        
                :return: A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]`.
                :rtype: str
        """
    def get_params(self) -> dict[str, OVAny]:
        """
                Returns a dict of device-specific parameters required for low-level
                operations with the underlying tensor.
                Parameters include device/context/surface/buffer handles, access flags, etc.
                Content of the returned dict depends on remote execution context that is
                currently set on the device (working scenario).
        
                :return: A dictionary of device-specific parameters.
                :rtype: dict
        """
    def get_shape(self) -> Shape:
        """
                Gets Tensor's shape.
        
                :rtype: openvino.Shape
        """
    @property
    def bytes_data(self) -> None:
        """
                This property is not implemented.
        """
    @bytes_data.setter
    def bytes_data(self, arg1: typing.Any) -> None:
        ...
    @property
    def data(self) -> None:
        """
                This property is not implemented.
        """
    @property
    def str_data(self) -> None:
        """
                This property is not implemented.
        """
    @str_data.setter
    def str_data(self, arg1: typing.Any) -> None:
        ...
class Shape:
    """
    openvino.Shape wraps ov::Shape
    """
    __hash__: typing.ClassVar[None] = None
    @typing.overload
    def __eq__(self, arg0: Shape) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: tuple) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: list) -> bool:
        ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __getitem__(self, arg0: slice) -> Shape:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, axis_lengths: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, axis_lengths: Shape) -> None:
        ...
    @typing.overload
    def __init__(self, shape: str) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Dimension) -> None:
        ...
    def __str__(self) -> str:
        ...
    def to_string(self) -> str:
        ...
class Strides:
    """
    openvino.Strides wraps ov::Strides
    """
    __hash__: typing.ClassVar[None] = None
    @typing.overload
    def __eq__(self, arg0: Strides) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: tuple) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: list) -> bool:
        ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __init__(self, axis_strides: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, axis_strides: Strides) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
class Symbol:
    """
    openvino.Symbol wraps ov::Symbol
    """
    def __bool__(self) -> bool:
        """
        Check whether the symbol is meaningful
        """
    def __eq__(self, arg0: Symbol) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self) -> None:
        ...
class TelemetryExtension(Extension):
    @typing.overload
    def __init__(self, arg0: str, arg1: collections.abc.Callable, arg2: collections.abc.Callable, arg3: collections.abc.Callable) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: collections.abc.Callable[[str, str, str, typing.SupportsInt], None], arg2: collections.abc.Callable[[str, str], None], arg3: collections.abc.Callable[[str, str], None]) -> None:
        ...
    def send_error(self, arg0: str) -> None:
        ...
    def send_event(self, arg0: str, arg1: str, arg2: typing.SupportsInt) -> None:
        ...
    def send_stack_trace(self, arg0: str) -> None:
        ...
class Tensor:
    """
    openvino.Tensor holding either copy of memory or shared host memory.
    """
    def __copy__(self) -> Tensor:
        ...
    def __deepcopy__(self, arg0: dict) -> Tensor:
        ...
    @typing.overload
    def __init__(self, array: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], shared_memory: bool = False) -> None:
        """
                        Tensor's special constructor.
        
                        :param array: Array to create the tensor from.
                        :type array: numpy.array
                        :param shared_memory: If `True`, this Tensor memory is being shared with a host.
                                              Any action performed on the host memory is reflected on this Tensor's memory!
                                              If `False`, data is being copied to this Tensor.
                                              Requires data to be C_CONTIGUOUS if `True`.
                                              If the passed array contains strings, the flag must be set to `False'.
                        :type shared_memory: bool
        """
    @typing.overload
    def __init__(self, array: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], shape: Shape, type: Type = ...) -> None:
        """
                        Another Tensor's special constructor.
        
                        Represents array in the memory with given shape and element type.
                        It's recommended to use this constructor only for wrapping array's
                        memory with the specific openvino element type parameter.
        
                        :param array: C_CONTIGUOUS numpy array which will be wrapped in
                                      openvino.Tensor with given parameters (shape
                                      and element_type). Array's memory is being shared with a host.
                                      Any action performed on the host memory will be reflected on this Tensor's memory!
                        :type array: numpy.array
                        :param shape: Shape of the new tensor.
                        :type shape: openvino.Shape
                        :param type: Element type
                        :type type: openvino.Type
        
                        :Example:
                        .. code-block:: python
        
                            import openvino as ov
                            import numpy as np
        
                            arr = np.array(shape=(100), dtype=np.uint8)
                            t = ov.Tensor(arr, ov.Shape([100, 8]), ov.Type.u1)
        """
    @typing.overload
    def __init__(self, array: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], shape: collections.abc.Sequence[typing.SupportsInt], type: Type = ...) -> None:
        """
                         Another Tensor's special constructor.
        
                        Represents array in the memory with given shape and element type.
                        It's recommended to use this constructor only for wrapping array's
                        memory with the specific openvino element type parameter.
        
                        :param array: C_CONTIGUOUS numpy array which will be wrapped in
                                      openvino.Tensor with given parameters (shape
                                      and element_type). Array's memory is being shared with a host.
                                      Any action performed on the host memory will be reflected on this Tensor's memory!
                        :type array: numpy.array
                        :param shape: Shape of the new tensor.
                        :type shape: list or tuple
                        :param type: Element type.
                        :type type: openvino.Type
        
                        :Example:
                        .. code-block:: python
        
                            import openvino as ov
                            import numpy as np
        
                            arr = np.array(shape=(100), dtype=np.uint8)
                            t = ov.Tensor(arr, [100, 8], ov.Type.u1)
        """
    @typing.overload
    def __init__(self, list: list) -> None:
        """
                        Tensor's special constructor.
        
                        Creates a Tensor from a given Python list.
                        Warning: It is always a copy of list's data!
        
                        :param array: list to create the tensor from.
                        :type array: list[int, float, str]
        """
    @typing.overload
    def __init__(self, type: Type, shape: Shape) -> None:
        ...
    @typing.overload
    def __init__(self, type: Type, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, type: numpy.dtype[typing.Any], shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, type: typing.Any, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, type: numpy.dtype[typing.Any], shape: Shape) -> None:
        ...
    @typing.overload
    def __init__(self, type: typing.Any, shape: Shape) -> None:
        ...
    @typing.overload
    def __init__(self, port: Output) -> None:
        """
                        Constructs Tensor using port from node.
                        Type and shape will be taken from the port.
        
                        :param port: Output port from a node.
                        :type param: openvino.Output
        """
    @typing.overload
    def __init__(self, port: Output, array: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]) -> None:
        """
                        Constructs Tensor using port from node.
                        Type and shape will be taken from the port.
        
                        :param port: Output port from a node.
                        :type param: openvino.Output
                        :param array: C_CONTIGUOUS numpy array which will be wrapped in
                                      openvino.Tensor. Array's memory is being shared wi a host.
                                      Any action performed on the host memory will be reflected on this Tensor's memory!
                        :type array: numpy.array
        """
    @typing.overload
    def __init__(self, port: typing.Any) -> None:
        """
                    Constructs Tensor using port from node.
                    Type and shape will be taken from the port.
        
                    :param port: Output port from a node.
                    :type param: openvino.ConstOutput
        """
    @typing.overload
    def __init__(self, port: typing.Any, array: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]) -> None:
        """
                        Constructs Tensor using port from node.
                        Type and shape will be taken from the port.
        
                        :param port: Output port from a node.
                        :type param: openvino.ConstOutput
                        :param array: C_CONTIGUOUS numpy array which will be wrapped in
                                      openvino.Tensor. Array's memory is being shared with a host.
                                      Any action performed on the host memory will be reflected on this Tensor's memory!
                        :type array: numpy.array
        """
    @typing.overload
    def __init__(self, other: Tensor, begin: Coordinate, end: Coordinate) -> None:
        ...
    @typing.overload
    def __init__(self, other: Tensor, begin: collections.abc.Sequence[typing.SupportsInt], end: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, image: typing.Any) -> None:
        """
                    Constructs Tensor from a Pillow Image.
        
                    :param image: Pillow Image to create the tensor from.
                    :type image: PIL.Image.Image
                    :Example:
                    .. code-block:: python
        
                        from PIL import Image
                        import openvino as ov
        
                        img = Image.open("example.jpg")
                        tensor = ov.Tensor(img)
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def copy_from(self, source_tensor: Tensor) -> None:
        """
                Copy source tensor's data to this tensor. Tensors should have the same element type and shape.
        
                :param source_tensor: The source tensor from which the data will be copied.
                :type source_tensor: openvino.Tensor
        """
    @typing.overload
    def copy_from(self, source_tensor: RemoteTensorWrapper) -> None:
        """
                Copy source remote tensor's data to this tensor. Tensors should have the same element type.
                In case of RoiTensor, tensors should also have the same shape.
        
                :param source_tensor: The source remote tensor from which the data will be copied.
                :type source_tensor: openvino.RemoteTensor
        """
    @typing.overload
    def copy_from(self, source: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]) -> None:
        """
                Copy the source to this tensor. This tensor and the source should have the same element type.
                Shape will be adjusted if there is a mismatch.
        """
    @typing.overload
    def copy_from(self, source: list) -> None:
        """
                Copy the source to this tensor. This tensor and the source should have the same element type.
                Shape will be adjusted if there is a mismatch.
        """
    @typing.overload
    def copy_to(self, target_tensor: Tensor) -> None:
        """
                Copy tensor's data to a destination tensor. The destination tensor should have the same element type and shape.
        
                :param target_tensor: The destination tensor to which the data will be copied.
                :type target_tensor: openvino.Tensor
        """
    @typing.overload
    def copy_to(self, target_tensor: RemoteTensorWrapper) -> None:
        """
                Copy tensor's data to a destination remote tensor. The destination remote tensor should have the same element type.
                In case of RoiRemoteTensor, the destination tensor should also have the same shape.
        
                :param target_tensor: The destination remote tensor to which the data will be copied.
                :type target_tensor: openvino.RemoteTensor
        """
    def get_byte_size(self) -> int:
        """
                    Gets Tensor's size in bytes.
        
                    :rtype: int
        """
    def get_element_type(self) -> Type:
        """
                    Gets Tensor's element type.
        
                    :rtype: openvino.Type
        """
    def get_shape(self) -> Shape:
        """
                    Gets Tensor's shape.
        
                    :rtype: openvino.Shape
        """
    def get_size(self) -> int:
        """
                    Gets Tensor's size as total number of elements.
        
                    :rtype: int
        """
    def get_strides(self) -> Strides:
        """
                    Gets Tensor's strides in bytes.
        
                    :rtype: openvino.Strides
        """
    def is_continuous(self) -> bool:
        """
                Reports whether the tensor is continuous or not.
                :return: True if the tensor is continuous, otherwise False.
                :rtype: bool
        """
    @typing.overload
    def set_shape(self, arg0: Shape) -> None:
        """
                    Sets Tensor's shape.
        """
    @typing.overload
    def set_shape(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
                    Sets Tensor's shape.
        """
    @property
    def byte_size(self) -> int:
        """
                                        Tensor's size in bytes.
        
                                        :rtype: int
        """
    @property
    def bytes_data(self) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        """
                    Access to Tensor's data with string Type in `np.bytes_` dtype.
        
                    Getter returns a numpy array with corresponding shape and dtype.
                    Warning: Data of string type is always a copy of underlaying memory!
        
                    Setter fills underlaying Tensor's memory by copying strings from `other`.
                    `other` must have the same size (number of elements) as the Tensor.
                    Tensor's shape is not changed by performing this operation!
        """
    @bytes_data.setter
    def bytes_data(self, arg1: typing.Any) -> None:
        ...
    @property
    def data(self) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        """
                    Access to Tensor's data.
        
                    Returns numpy array with corresponding shape and dtype.
        
                    For tensors with OpenVINO specific element type, such as u1, u4 or i4
                    it returns linear array, with uint8 / int8 numpy dtype.
        
                    For tensors with string element type, returns a numpy array of bytes
                    without any decoding.
                    To change the underlaying data use `str_data`/`bytes_data` properties
                    or the `copy_from` function.
                    Warning: Data of string type is always a copy of underlaying memory!
        
                    :rtype: numpy.array
        """
    @property
    def element_type(self) -> Type:
        """
                                        Tensor's element type.
        
                                        :rtype: openvino.Type
        """
    @property
    def shape(self) -> Shape:
        """
                    Tensor's shape get/set.
        """
    @shape.setter
    def shape(self, arg1: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @property
    def size(self) -> int:
        """
                                        Tensor's size as total number of elements.
        
                                        :rtype: int
        """
    @property
    def str_data(self) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        """
                    Access to Tensor's data with string Type in `np.str_` dtype.
        
                    Getter returns a numpy array with corresponding shape and dtype.
                    Warning: Data of string type is always a copy of underlaying memory!
        
                    Setter fills underlaying Tensor's memory by copying strings from `other`.
                    `other` must have the same size (number of elements) as the Tensor.
                    Tensor's shape is not changed by performing this operation!
        """
    @str_data.setter
    def str_data(self, arg1: typing.Any) -> None:
        ...
    @property
    def strides(self) -> Strides:
        """
                                        Tensor's strides in bytes.
        
                                        :rtype: openvino.Strides
        """
class Type:
    """
    openvino.Type wraps ov::element::Type
    """
    bf16: typing.ClassVar[Type]  # value = <Type: 'bfloat16'>
    boolean: typing.ClassVar[Type]  # value = <Type: 'char'>
    dynamic: typing.ClassVar[Type]  # value = <Type: 'dynamic'>
    f16: typing.ClassVar[Type]  # value = <Type: 'float16'>
    f32: typing.ClassVar[Type]  # value = <Type: 'float32'>
    f4e2m1: typing.ClassVar[Type]  # value = <Type: 'f4e2m1'>
    f64: typing.ClassVar[Type]  # value = <Type: 'double64'>
    f8e4m3: typing.ClassVar[Type]  # value = <Type: 'f8e4m3'>
    f8e5m2: typing.ClassVar[Type]  # value = <Type: 'f8e5m2'>
    f8e8m0: typing.ClassVar[Type]  # value = <Type: 'f8e8m0'>
    i16: typing.ClassVar[Type]  # value = <Type: 'int16_t'>
    i32: typing.ClassVar[Type]  # value = <Type: 'int32_t'>
    i4: typing.ClassVar[Type]  # value = <Type: 'int4_t'>
    i64: typing.ClassVar[Type]  # value = <Type: 'int64_t'>
    i8: typing.ClassVar[Type]  # value = <Type: 'int8_t'>
    nf4: typing.ClassVar[Type]  # value = <Type: 'nfloat4'>
    string: typing.ClassVar[Type]  # value = <Type: 'string'>
    u1: typing.ClassVar[Type]  # value = <Type: 'uint1_t'>
    u16: typing.ClassVar[Type]  # value = <Type: 'uint16_t'>
    u32: typing.ClassVar[Type]  # value = <Type: 'uint32_t'>
    u4: typing.ClassVar[Type]  # value = <Type: 'uint4_t'>
    u64: typing.ClassVar[Type]  # value = <Type: 'uint64_t'>
    u8: typing.ClassVar[Type]  # value = <Type: 'uint8_t'>
    undefined: typing.ClassVar[Type]  # value = <Type: 'dynamic'>
    def __eq__(self, arg0: Type) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self, dtype: typing.Any) -> None:
        """
                    Convert numpy dtype into OpenVINO type
        
                    :param dtype: numpy dtype
                    :type dtype: numpy.dtype
                    :return: OpenVINO type object
                    :rtype: openvino.Type
        """
    def __repr__(self) -> str:
        ...
    def compatible(self, other: Type) -> bool:
        """
                        Checks whether this element type is merge-compatible with
                        `other`.
        
                        :param other: The element type to compare this element type to.
                        :type other: openvino.Type
                        :return: `True` if element types are compatible, otherwise `False`.
                        :rtype: bool
        """
    def get_bitwidth(self) -> int:
        ...
    def get_size(self) -> int:
        ...
    def get_type_name(self) -> str:
        ...
    def is_dynamic(self) -> bool:
        ...
    def is_integral(self) -> bool:
        ...
    def is_integral_number(self) -> bool:
        ...
    def is_quantized(self) -> bool:
        ...
    def is_real(self) -> bool:
        ...
    def is_signed(self) -> bool:
        ...
    def is_static(self) -> bool:
        ...
    def merge(self, other: Type) -> typing.Any:
        """
                    Merge two element types and return result if successful,
                    otherwise return None.
        
                    :param other: The element type to merge with this element type.
                    :type other: openvino.Type
                    :return: If element types are compatible return the least
                             restrictive Type, otherwise `None`.
                    :rtype: Union[openvino.Type|None]
        """
    def to_dtype(self) -> numpy.dtype[typing.Any]:
        """
                    Convert Type to numpy dtype.
        
                    :return: dtype object
                    :rtype: numpy.dtype
        """
    def to_string(self) -> str:
        ...
    @property
    def bitwidth(self) -> int:
        ...
    @property
    def integral(self) -> bool:
        ...
    @property
    def integral_number(self) -> bool:
        ...
    @property
    def quantized(self) -> bool:
        ...
    @property
    def real(self) -> bool:
        ...
    @property
    def signed(self) -> bool:
        ...
    @property
    def size(self) -> int:
        ...
    @property
    def type_name(self) -> str:
        ...
class VAContext(RemoteContext):
    def __init__(self, core: Core, display: typing_extensions.CapsuleType, target_tile_id: typing.SupportsInt = -1) -> None:
        """
                    Constructs remote context object from valid VA display handle.
        
                    :param core: OpenVINO Runtime Core object.
                    :type core: openvino.Core
                    :param device: A valid `VADisplay` to create remote context from.
                    :type device: Any
                    :param target_tile_id: Desired tile id within given context for multi-tile system.
                                           Default value (-1) means that root device should be used.
                    :type target_tile_id: int
                    :return: A context instance.
                    :rtype: openvino.VAContext
        """
    def create_tensor(self, type: Type, shape: Shape, surface: typing.SupportsInt, plane: typing.SupportsInt = 0) -> VASurfaceTensorWrapper:
        """
                    Create remote tensor from VA surface handle.
        
                    GIL is released while running this function.
        
                    :param type: Defines the element type of the tensor.
                    :type type: openvino.Type
                    :param shape: Defines the shape of the tensor.
                    :type shape: openvino.Shape
                    :param surface: `VASurfaceID` to create tensor from.
                    :type surface: int
                    :param plane: An index of a plane inside `VASurfaceID` to create tensor from. Default: 0
                    :type plane: int
                    :return: A remote tensor instance wrapping `VASurfaceID`.
                    :rtype: openvino.VASurfaceTensor
        """
    def create_tensor_nv12(self, height: typing.SupportsInt, width: typing.SupportsInt, nv12_surface: typing.SupportsInt) -> tuple:
        """
                    This function is used to obtain a NV12 tensor from NV12 VA decoder output.
                    The result contains two remote tensors for Y and UV planes of the surface.
        
                    GIL is released while running this function.
        
                    :param height: A height of Y plane.
                    :type height: int
                    :param width: A width of Y plane
                    :type width: int
                    :param nv12_surface: NV12 `VASurfaceID` to create NV12 from.
                    :type nv12_surface: int
                    :return: A pair of remote tensors for each plane.
                    :rtype: tuple[openvino.VASurfaceTensor, openvino.VASurfaceTensor]
        """
class VASurfaceTensor(RemoteTensor):
    def __repr__(self) -> str:
        ...
    @property
    def data(self) -> None:
        """
                This property is not implemented.
        """
    @property
    def plane_id(self) -> int:
        """
                Returns plane ID of underlying video decoder surface.
        
                :return: Plane ID of underlying video decoder surface.
                :rtype: int
        """
    @property
    def surface_id(self) -> int:
        """
                Returns ID of underlying video decoder surface.
        
                :return: VASurfaceID of the tensor.
                :rtype: int
        """
class VariableState:
    """
    openvino.VariableState class.
    """
    def __repr__(self) -> str:
        ...
    def reset(self) -> None:
        """
                Reset internal variable state for relevant infer request,
                to a value specified as default for according node.
        """
    @property
    def name(self) -> str:
        """
                Gets name of current variable state.
        
                :return: A string representing a state name.
                :rtype: str
        """
    @property
    def state(self) -> Tensor:
        """
                Gets/sets variable state.
        """
    @state.setter
    def state(self, arg1: Tensor) -> None:
        ...
class Version:
    """
    openvino.Version represents version information that describes plugins and the OpenVINO library.
    """
    def __repr__(self) -> str:
        ...
    @property
    def build_number(self) -> str:
        """
                                :return: String with build number.
                                :rtype: str
        """
    @property
    def description(self) -> str:
        """
                                :return: Description string.
                                :rtype: str
        """
    @property
    def major(self) -> int:
        """
                    :return: OpenVINO's major version.
                    :rtype: int
        """
    @property
    def minor(self) -> int:
        """
                    :return: OpenVINO's minor version.
                    :rtype: int
        """
    @property
    def patch(self) -> int:
        """
                    :return: OpenVINO's version patch.
                    :rtype: int
        """
class _ConversionExtension(ConversionExtensionBase):
    pass
class _IDecoder:
    pass
@typing.overload
def get_batch(arg0: Model) -> Dimension:
    ...
@typing.overload
def get_batch(model: typing.Any) -> Dimension:
    ...
def get_version() -> str:
    ...
def save_model(model: typing.Any, output_model: typing.Any, compress_to_fp16: bool = True) -> None:
    """
                Save model into IR files (xml and bin). Floating point weights are compressed to FP16 by default.
                This method saves a model to IR applying all necessary transformations that usually applied
                in model conversion flow provided by OVC tool. Paricularly, floatting point weights are
                compressed to FP16, debug information in model nodes are cleaned up, etc.
    
                :param model: model which will be converted to IR representation
                :type model: openvino.Model
                :param output_model: path to output model file
                :type output_model: Union[str, bytes, pathlib.Path]
                :param compress_to_fp16: whether to compress floating point weights to FP16 (default: True). The parameter is ignored for pre-optimized models.
                :type compress_to_fp16: bool
    
                :Examples:
    
                .. code-block:: python
    
                    model = convert_model('your_model.onnx')
                    save_model(model, './model.xml')
    """
def serialize(model: typing.Any, xml_path: typing.Any, bin_path: typing.Any = '', version: str = 'UNSPECIFIED') -> None:
    """
                Serialize given model into IR. The generated .xml and .bin files will be saved
                into provided paths.
                This method serializes model "as-is" that means no weights compression is applied.
                It is recommended to use ov::save_model function instead of ov::serialize in all cases
                when it is not related to debugging.
    
                :param model: model which will be converted to IR representation
                :type model: openvino.Model
                :param xml_path: path where .xml file will be saved
                :type xml_path: Union[str, bytes, pathlib.Path]
                :param bin_path: path where .bin file will be saved (optional),
                                 the same name as for xml_path will be used by default.
                :type bin_path: Union[str, bytes, pathlib.Path]
                :param version: version of the generated IR (optional).
                :type version: str
    
                Supported versions are:
                - "UNSPECIFIED" (default) : Use the latest or model version
                - "IR_V10" : v10 IR
                - "IR_V11" : v11 IR
    
                :Examples:
    
                1. Default IR version:
    
                .. code-block:: python
    
                    shape = [2, 2]
                    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
                    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
                    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
                    op = (parameter_a + parameter_b) * parameter_c
                    model = Model(op, [parameter_a, parameter_b, parameter_c], "Model")
                    # IR generated with default version
                    serialize(model, xml_path="./serialized.xml", bin_path="./serialized.bin")
                2. IR version 11:
    
                .. code-block:: python
    
                    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
                    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
                    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
                    op = (parameter_a + parameter_b) * parameter_c
                    model = Model(ops, [parameter_a, parameter_b, parameter_c], "Model")
                    # IR generated with default version
                    serialize(model, xml_path="./serialized.xml", bin_path="./serialized.bin", version="IR_V11")
    """
@typing.overload
def set_batch(model: typing.Any, dimension: Dimension) -> None:
    ...
@typing.overload
def set_batch(model: typing.Any, batch_size: typing.SupportsInt = -1) -> None:
    ...
def shutdown() -> None:
    """
                        Shut down the OpenVINO by deleting all static-duration objects allocated by the library and releasing
                        dependent resources
    
                        This function should be used by advanced user to control unload the resources.
    
                        You might want to use this function if you are developing a dynamically-loaded library which should clean up all
                        resources after itself when the library is unloaded.
    """
