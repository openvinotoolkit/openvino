# type: ignore
from __future__ import annotations
from builtins import traceback as TracebackType
from collections.abc import Iterator
from openvino._pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino._pyopenvino import CompiledModel as CompiledModelBase
from openvino._pyopenvino import Core as CoreBase
from openvino._pyopenvino import Model as ModelBase
from openvino._pyopenvino import Node
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import Type
from openvino.package_utils import deprecatedclassproperty
from openvino.utils.data_helpers.data_dispatcher import _data_dispatch
from openvino.utils.data_helpers.wrappers import OVDict
from openvino.utils.data_helpers.wrappers import _InferRequestWrapper
from openvino.utils.data_helpers.wrappers import tensor_from_file
from pathlib import Path
import collections.abc
import io as io
import openvino._pyopenvino
import openvino._pyopenvino.op
import openvino._pyopenvino.op.util
import openvino.utils.data_helpers.wrappers
import pathlib
import traceback as traceback
import typing
__all__ = ['AsyncInferQueue', 'AsyncInferQueueBase', 'CompiledModel', 'CompiledModelBase', 'Core', 'CoreBase', 'InferRequest', 'Iterator', 'Model', 'ModelBase', 'ModelMeta', 'Node', 'OVDict', 'Path', 'Tensor', 'TracebackType', 'Type', 'compile_model', 'deprecatedclassproperty', 'io', 'tensor_from_file', 'traceback']
class AsyncInferQueue(openvino._pyopenvino.AsyncInferQueue):
    """
    AsyncInferQueue with a pool of asynchronous requests.
    
        AsyncInferQueue represents a helper that creates a pool of asynchronous
        InferRequests and provides synchronization functions to control flow of
        a simple pipeline.
        
    """
    def __getitem__(self, i: int) -> InferRequest:
        """
        Gets InferRequest from the pool with given i id.
        
                Resulting object is guaranteed to work with read-only methods like getting tensors.
                Any mutating methods (e.g. start_async, set_callback) of a request
                will put the parent AsyncInferQueue object in an invalid state.
        
                :param i:  InferRequest id.
                :type i: int
                :return: InferRequests from the pool with given id.
                :rtype: openvino.InferRequest
                
        """
    def __iter__(self) -> collections.abc.Iterator[InferRequest]:
        """
        Allows to iterate over AsyncInferQueue.
        
                Resulting objects are guaranteed to work with read-only methods like getting tensors.
                Any mutating methods (e.g. start_async, set_callback) of a single request
                will put the parent AsyncInferQueue object in an invalid state.
        
                :return: a generator that yields InferRequests.
                :rtype: collections.abc.Iterable[openvino.InferRequest]
                
        """
    def start_async(self, inputs: typing.Any = None, userdata: typing.Any = None, share_inputs: bool = False) -> None:
        """
        Run asynchronous inference using the next available InferRequest from the pool.
        
                The allowed types of keys in the `inputs` dictionary are:
        
                (1) `int`
                (2) `str`
                (3) `openvino.ConstOutput`
        
                The allowed types of values in the `inputs` are:
        
                (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                (2) `openvino.Tensor`
        
                Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
                it will work only with one-input models. When model has more inputs,
                function throws error.
        
                :param inputs: Data to be set on input tensors of the next available InferRequest.
                :type inputs: Any, optional
                :param userdata: Any data that will be passed to a callback.
                :type userdata: Any, optional
                :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.
        
                                      If set to `False` inputs the data dispatcher will safely copy data
                                      to existing Tensors (including up- or down-casting according to data type,
                                      resizing of the input Tensor). Keeps Tensor inputs "as-is".
        
                                      If set to `True` the data dispatcher tries to provide "zero-copy"
                                      Tensors for every input in form of:
                                      * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                                      Data that is going to be copied:
                                      * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                                      * inputs which data types are mismatched from Infer Request's inputs
                                      * inputs that should be in `BF16` data type
                                      * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                                      * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                                      Keeps Tensor inputs "as-is".
        
                                      Note: Use with extra care, shared data can be modified during runtime!
                                      Note: Using `share_inputs` may result in extra memory overhead.
        
                                      Default value: False
                :type share_inputs: bool, optional
                
        """
class CompiledModel(openvino._pyopenvino.CompiledModel):
    """
    CompiledModel class.
    
        CompiledModel represents Model that is compiled for a specific device by applying
        multiple optimization transformations, then mapping to compute kernels.
        
    """
    def __call__(self, inputs: typing.Any = None, share_inputs: bool = True, share_outputs: bool = False, *, decode_strings: bool = True) -> openvino.utils.data_helpers.wrappers.OVDict:
        """
        Callable infer wrapper for CompiledModel.
        
                Infers specified input(s) in synchronous mode.
        
                Blocks all methods of CompiledModel while request is running.
        
                Method creates new temporary InferRequest and run inference on it.
                It is advised to use a dedicated InferRequest class for performance,
                optimizing workflows, and creating advanced pipelines.
        
                This method stores created `InferRequest` inside `CompiledModel` object,
                which can be later reused in consecutive calls.
        
                The allowed types of keys in the `inputs` dictionary are:
        
                (1) `int`
                (2) `str`
                (3) `openvino.ConstOutput`
        
                The allowed types of values in the `inputs` are:
        
                (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                (2) `openvino.Tensor`
        
                Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
                it will work only with one-input models. When model has more inputs,
                function throws error.
        
                :param inputs: Data to be set on input tensors.
                :type inputs: Any, optional
                :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.
        
                                      If set to `False` inputs the data dispatcher will safely copy data
                                      to existing Tensors (including up- or down-casting according to data type,
                                      resizing of the input Tensor). Keeps Tensor inputs "as-is".
        
                                      If set to `True` the data dispatcher tries to provide "zero-copy"
                                      Tensors for every input in form of:
                                      * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                                      Data that is going to be copied:
                                      * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                                      * inputs which data types are mismatched from Infer Request's inputs
                                      * inputs that should be in `BF16` data type
                                      * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                                      * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                                      Keeps Tensor inputs "as-is".
        
                                      Note: Use with extra care, shared data can be modified during runtime!
                                      Note: Using `share_inputs` may result in extra memory overhead.
        
                                      Default value: True
                :type share_inputs: bool, optional
                :param share_outputs: Enables `share_outputs` mode. Controls memory usage on inference's outputs.
        
                                      If set to `False` outputs will safely copy data to numpy arrays.
        
                                      If set to `True` the data will be returned in form of views of output Tensors.
                                      This mode still returns the data in format of numpy arrays but lifetime of the data
                                      is connected to OpenVINO objects.
        
                                      Note: Use with extra care, shared data can be modified or lost during runtime!
                                      Note: String/textual data will always be copied!
        
                                      Default value: False
                :type share_outputs: bool, optional
                :param decode_strings: Controls decoding outputs of textual based data.
        
                                       If set to `True` string outputs will be returned as numpy arrays of `U` kind.
        
                                       If set to `False` string outputs will be returned as numpy arrays of `S` kind.
        
                                       Default value: True
                :type decode_strings: bool, optional, keyword-only
        
                :return: Dictionary of results from output tensors with port/int/str as keys.
                :rtype: OVDict
                
        """
    def __init__(self, other: openvino._pyopenvino.CompiledModel, weights: typing.Optional[bytes] = None) -> None:
        ...
    def create_infer_request(self) -> InferRequest:
        """
        Creates an inference request object used to infer the compiled model.
        
                The created request has allocated input and output tensors.
        
                :return: New InferRequest object.
                :rtype: openvino.InferRequest
                
        """
    def get_runtime_model(self) -> Model:
        ...
    def infer_new_request(self, inputs: typing.Any = None) -> openvino.utils.data_helpers.wrappers.OVDict:
        """
        Infers specified input(s) in synchronous mode.
        
                Blocks all methods of CompiledModel while request is running.
        
                Method creates new temporary InferRequest and run inference on it.
                It is advised to use a dedicated InferRequest class for performance,
                optimizing workflows, and creating advanced pipelines.
        
                The allowed types of keys in the `inputs` dictionary are:
        
                (1) `int`
                (2) `str`
                (3) `openvino.ConstOutput`
        
                The allowed types of values in the `inputs` are:
        
                (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                (2) `openvino.Tensor`
        
                Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
                it will work only with one-input models. When model has more inputs,
                function throws error.
        
                :param inputs: Data to be set on input tensors.
                :type inputs: Any, optional
                :return: Dictionary of results from output tensors with port/int/str keys.
                :rtype: OVDict
                
        """
    def query_state(self) -> None:
        """
        Gets state control interface for the underlaying infer request.
        
                :return: list of VariableState objects.
                :rtype: list[openvino.VariableState]
                
        """
    def reset_state(self) -> None:
        """
        Resets all internal variable states of the underlaying infer request.
        
                Resets all internal variable states to a value specified as default for
                the corresponding `ReadValue` node.
                
        """
class Core(openvino._pyopenvino.Core):
    """
    Core class represents OpenVINO runtime Core entity.
    
        User applications can create several Core class instances, but in this
        case, the underlying plugins are created multiple times and not shared
        between several Core instances. The recommended way is to have a single
        Core instance per application.
        
    """
    def compile_model(self, model: typing.Union[openvino._ov_api.Model, str, pathlib.Path], device_name: typing.Optional[str] = None, config: typing.Optional[dict[str, typing.Any]] = None, *, weights: typing.Optional[bytes] = None) -> CompiledModel:
        """
        Creates a compiled model.
        
                Creates a compiled model from a source Model object or
                reads model and creates a compiled model from IR / ONNX / PDPD / TF and TFLite file or
                creates a compiled model from a IR xml and weights in memory.
                This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
                especially for cases when caching is enabled and cached model is available.
                If device_name is not specified, the default OpenVINO device will be selected by AUTO plugin.
                Users can create as many compiled models as they need, and use them simultaneously
                (up to the limitation of the hardware resources).
        
                :param model: Model acquired from read_model function or a path to a model in IR / ONNX / PDPD /
                              TF and TFLite format.
                :type model: Union[openvino.Model, str, pathlib.Path]
                :param device_name: Optional. Name of the device to load the model to. If not specified,
                                    the default OpenVINO device will be selected by AUTO plugin.
                :type device_name: str
                :param config: Optional dict of pairs:
                               (property name, property value) relevant only for this load operation.
                :type config: dict, optional
                :param weights: Optional. Weights of model in memory to be loaded to the model.
                :type weights: bytes, optional, keyword-only
                :return: A compiled model.
                :rtype: openvino.CompiledModel
                
        """
    def import_model(self, model_stream: bytes, device_name: str, config: typing.Optional[dict[str, typing.Any]] = None) -> CompiledModel:
        """
        Imports a compiled model from a previously exported one.
        
                :param model_stream: Input stream, containing a model previously exported, using export_model method.
                :type model_stream: bytes
                :param device_name: Name of device to which compiled model is imported.
                                    Note: if device_name is not used to compile the original model,
                                    an exception is thrown.
                :type device_name: str
                :param config: Optional dict of pairs:
                               (property name, property value) relevant only for this load operation.
                :type config: dict, optional
                :return: A compiled model.
                :rtype: openvino.CompiledModel
        
                :Example:
        
                .. code-block:: python
        
                    user_stream = compiled.export_model()
        
                    with open('./my_model', 'wb') as f:
                        f.write(user_stream)
        
                    # ...
        
                    new_compiled = core.import_model(user_stream, "CPU")
        
                .. code-block:: python
        
                    user_stream = io.BytesIO()
                    compiled.export_model(user_stream)
        
                    with open('./my_model', 'wb') as f:
                        f.write(user_stream.getvalue()) # or read() if seek(0) was applied before
        
                    # ...
        
                    new_compiled = core.import_model(user_stream, "CPU")
                
        """
    def query_model(self, model: Model, device_name: str, config: typing.Optional[dict[str, typing.Any]] = None) -> dict:
        ...
    def read_model(self, model: typing.Union[str, bytes, object, _io.BytesIO], weights: typing.Union[object, str, bytes, openvino._pyopenvino.Tensor, _io.BytesIO] = None, config: typing.Optional[dict[str, typing.Any]] = None) -> Model:
        ...
class InferRequest(openvino.utils.data_helpers.wrappers._InferRequestWrapper):
    """
    InferRequest class represents infer request which can be run in asynchronous or synchronous manners.
    """
    def get_compiled_model(self) -> CompiledModel:
        """
        Gets the compiled model this InferRequest is using.
        
                :return: a CompiledModel object
                :rtype: openvino.CompiledModel
                
        """
    def infer(self, inputs: typing.Any = None, share_inputs: bool = False, share_outputs: bool = False, *, decode_strings: bool = True) -> openvino.utils.data_helpers.wrappers.OVDict:
        """
        Infers specified input(s) in synchronous mode.
        
                Blocks all methods of InferRequest while request is running.
                Calling any method will lead to throwing exceptions.
        
                The allowed types of keys in the `inputs` dictionary are:
        
                (1) `int`
                (2) `str`
                (3) `openvino.ConstOutput`
        
                The allowed types of values in the `inputs` are:
        
                (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                (2) `openvino.Tensor`
        
                Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
                it will work only with one-input models. When model has more inputs,
                function throws error.
        
                :param inputs: Data to be set on input tensors.
                :type inputs: Any, optional
                :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.
        
                                      If set to `False` inputs the data dispatcher will safely copy data
                                      to existing Tensors (including up- or down-casting according to data type,
                                      resizing of the input Tensor). Keeps Tensor inputs "as-is".
        
                                      If set to `True` the data dispatcher tries to provide "zero-copy"
                                      Tensors for every input in form of:
                                      * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                                      Data that is going to be copied:
                                      * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                                      * inputs which data types are mismatched from Infer Request's inputs
                                      * inputs that should be in `BF16` data type
                                      * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                                      * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                                      Keeps Tensor inputs "as-is".
        
                                      Note: Use with extra care, shared data can be modified during runtime!
                                      Note: Using `share_inputs` may result in extra memory overhead.
        
                                      Default value: False
                :type share_inputs: bool, optional
                :param share_outputs: Enables `share_outputs` mode. Controls memory usage on inference's outputs.
        
                                      If set to `False` outputs will safely copy data to numpy arrays.
        
                                      If set to `True` the data will be returned in form of views of output Tensors.
                                      This mode still returns the data in format of numpy arrays but lifetime of the data
                                      is connected to OpenVINO objects.
        
                                      Note: Use with extra care, shared data can be modified or lost during runtime!
                                      Note: String/textual data will always be copied!
        
                                      Default value: False
                :type share_outputs: bool, optional
                :param decode_strings: Controls decoding outputs of textual based data.
        
                                       If set to `True` string outputs will be returned as numpy arrays of `U` kind.
        
                                       If set to `False` string outputs will be returned as numpy arrays of `S` kind.
        
                                       Default value: True
                :type decode_strings: bool, optional, keyword-only
        
                :return: Dictionary of results from output tensors with port/int/str keys.
                :rtype: OVDict
                
        """
    def start_async(self, inputs: typing.Any = None, userdata: typing.Any = None, share_inputs: bool = False) -> None:
        """
        Starts inference of specified input(s) in asynchronous mode.
        
                Returns immediately. Inference starts also immediately.
                Calling any method on the `InferRequest` object while the request is running
                will lead to throwing exceptions.
        
                The allowed types of keys in the `inputs` dictionary are:
        
                (1) `int`
                (2) `str`
                (3) `openvino.ConstOutput`
        
                The allowed types of values in the `inputs` are:
        
                (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                (2) `openvino.Tensor`
        
                Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
                it will work only with one-input models. When model has more inputs,
                function throws error.
        
                :param inputs: Data to be set on input tensors.
                :type inputs: Any, optional
                :param userdata: Any data that will be passed inside the callback.
                :type userdata: Any
                :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.
        
                                      If set to `False` inputs the data dispatcher will safely copy data
                                      to existing Tensors (including up- or down-casting according to data type,
                                      resizing of the input Tensor). Keeps Tensor inputs "as-is".
        
                                      If set to `True` the data dispatcher tries to provide "zero-copy"
                                      Tensors for every input in form of:
                                      * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                                      Data that is going to be copied:
                                      * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                                      * inputs which data types are mismatched from Infer Request's inputs
                                      * inputs that should be in `BF16` data type
                                      * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                                      * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                                      Keeps Tensor inputs "as-is".
        
                                      Note: Use with extra care, shared data can be modified during runtime!
                                      Note: Using `share_inputs` may result in extra memory overhead.
        
                                      Default value: False
                :type share_inputs: bool, optional
                
        """
    @property
    def results(self) -> openvino.utils.data_helpers.wrappers.OVDict:
        """
        Gets all outputs tensors of this InferRequest.
        
                :return: Dictionary of results from output tensors with ports as keys.
                :rtype: dict[openvino.ConstOutput, numpy.array]
                
        """
class Model:
    def __copy__(self) -> Model:
        ...
    def __deepcopy__(self, memo: dict) -> Model:
        """
        Returns a deepcopy of Model.
        
                :return: A copy of Model.
                :rtype: openvino.Model
                
        """
    def __enter__(self) -> Model:
        ...
    def __exit__(self, exc_type: type[BaseException], exc_value: BaseException, traceback: traceback) -> None:
        ...
    def __getattr__(self, name: str) -> typing.Any:
        ...
    @typing.overload
    def __init__(self: openvino._pyopenvino.Model, other: openvino._pyopenvino.Model) -> None:
        ...
    @typing.overload
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.op.Result], sinks: collections.abc.Sequence[openvino._pyopenvino.Node], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.op.Result], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Node], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, result: openvino._pyopenvino.Node, parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Output], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Output], sinks: collections.abc.Sequence[openvino._pyopenvino.Node], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Output], sinks: collections.abc.Sequence[openvino._pyopenvino.Output], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Output], sinks: collections.abc.Sequence[openvino._pyopenvino.Output], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], variables: collections.abc.Sequence[openvino._pyopenvino.op.util.Variable], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.op.Result], sinks: collections.abc.Sequence[openvino._pyopenvino.Output], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.op.Result], sinks: collections.abc.Sequence[openvino._pyopenvino.Output], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], variables: collections.abc.Sequence[openvino._pyopenvino.op.util.Variable], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.op.Result], sinks: collections.abc.Sequence[openvino._pyopenvino.Node], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], variables: collections.abc.Sequence[openvino._pyopenvino.op.util.Variable], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Output], sinks: collections.abc.Sequence[openvino._pyopenvino.Node], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], variables: collections.abc.Sequence[openvino._pyopenvino.op.util.Variable], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.op.Result], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], variables: collections.abc.Sequence[openvino._pyopenvino.op.util.Variable], name: str = '') -> None:
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
    def __init__(self: openvino._pyopenvino.Model, results: collections.abc.Sequence[openvino._pyopenvino.Output], parameters: collections.abc.Sequence[openvino._pyopenvino.op.Parameter], variables: collections.abc.Sequence[openvino._pyopenvino.op.util.Variable], name: str = '') -> None:
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
    def clone(self) -> Model:
        ...
class ModelMeta(type):
    @classmethod
    def __getattr__(cls, name: str) -> typing.Any:
        ...
    @classmethod
    def __getattribute__(cls, name: str) -> typing.Any:
        ...
def compile_model(model: typing.Union[openvino._ov_api.Model, str, pathlib.Path], device_name: typing.Optional[str] = 'AUTO', config: typing.Optional[dict[str, typing.Any]] = None) -> CompiledModel:
    """
    Compact method to compile model with AUTO plugin.
    
        :param model: Model acquired from read_model function or a path to a model in IR / ONNX / PDPD /
                        TF and TFLite format.
        :type model: Union[openvino.Model, str, pathlib.Path]
        :param device_name: Optional. Name of the device to load the model to. If not specified,
                            the default OpenVINO device will be selected by AUTO plugin.
        :type device_name: str
        :param config: Optional dict of pairs:
                        (property name, property value) relevant only for this load operation.
        :type config: dict, optional
        :return: A compiled model.
        :rtype: openvino.CompiledModel
    
        
    """
