# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Any, Union

from openvino.pyopenvino import Model
from openvino.pyopenvino import Core as CoreBase
from openvino.pyopenvino import CompiledModel as CompiledModelBase
from openvino.pyopenvino import InferRequest as InferRequestBase
from openvino.pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino.pyopenvino import ConstOutput
from openvino.pyopenvino import Tensor
from openvino.pyopenvino import OVAny as OVAnyBase

from openvino.runtime.utils.types import get_dtype


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))


def convert_dict_items(inputs: dict, py_types: dict) -> dict:
    """Helper function converting dictionary items to Tensors."""
    # Create new temporary dictionary.
    # new_inputs will be used to transfer data to inference calls,
    # ensuring that original inputs are not overwritten with Tensors.
    new_inputs = {}
    for k, val in inputs.items():
        if not isinstance(k, (str, int, ConstOutput)):
            raise TypeError("Incompatible key type for tensor: {}".format(k))
        try:
            ov_type = py_types[k]
        except KeyError:
            raise KeyError("Port for tensor {} was not found!".format(k))
        # Convert numpy arrays or copy Tensors
        new_inputs[k] = (
            val
            if isinstance(val, Tensor)
            else Tensor(np.array(val, get_dtype(ov_type), copy=False))
        )
    return new_inputs


def normalize_inputs(inputs: Union[dict, list], py_types: dict) -> dict:
    """Normalize a dictionary of inputs to Tensors."""
    if isinstance(inputs, dict):
        return convert_dict_items(inputs, py_types)
    elif isinstance(inputs, list):
        # Lists are required to be represented as dictionaries with int keys
        return convert_dict_items(
            {index: input for index, input in enumerate(inputs)}, py_types
        )
    else:
        raise TypeError(
            "Inputs should be either list or dict! Current type: {}".format(
                type(inputs)
            )
        )


def get_input_types(obj: Union[InferRequestBase, CompiledModelBase]) -> dict:
    """Map all tensor names of all inputs to the data types of those tensors."""

    def get_inputs(obj: Union[InferRequestBase, CompiledModelBase]) -> list:
        return obj.model_inputs if isinstance(obj, InferRequestBase) else obj.inputs

    def map_tensor_names_to_types(input: ConstOutput) -> dict:
        return {n: input.get_element_type() for n in input.get_names()}

    input_types: dict = {}
    for idx, input in enumerate(get_inputs(obj)):
        # Add all possible "accessing aliases" to dictionary
        # Key as a ConstOutput port
        input_types[input] = input.get_element_type()
        # Key as an integer
        input_types[idx] = input.get_element_type()
        # Multiple possible keys as Tensor names
        input_types.update(map_tensor_names_to_types(input))
    return input_types


class InferRequest(InferRequestBase):
    """InferRequest class represents infer request which can be run in asynchronous or synchronous manners."""

    def infer(self, inputs: Union[dict, list] = None) -> dict:
        """Infers specified input(s) in synchronous mode.

        Blocks all methods of InferRequest while request is running.
        Calling any method will lead to throwing exceptions.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.runtime.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.array`
        (2) `openvino.runtime.Tensor`

        :param inputs: Data to be set on input tensors.
        :type inputs: Union[Dict[keys, values], List[values]], optional
        :return: Dictionary of results from output tensors with ports as keys.
        :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        """
        return super().infer(
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )

    def start_async(
        self, inputs: Union[dict, list] = None, userdata: Any = None
    ) -> None:
        """Starts inference of specified input(s) in asynchronous mode.

        Returns immediately. Inference starts also immediately.
        Calling any method on the `InferRequest` object while the request is running
        will lead to throwing exceptions.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.runtime.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.array`
        (2) `openvino.runtime.Tensor`

        :param inputs: Data to be set on input tensors.
        :type inputs: Union[Dict[keys, values], List[values]], optional
        :param userdata: Any data that will be passed inside the callback.
        :type userdata: Any
        """
        super().start_async(
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self)),
            userdata,
        )


class CompiledModel(CompiledModelBase):
    """CompiledModel class.

    CompiledModel represents Model that is compiled for a specific device by applying
    multiple optimization transformations, then mapping to compute kernels.
    """

    def create_infer_request(self) -> InferRequest:
        """Creates an inference request object used to infer the compiled model.

        The created request has allocated input and output tensors.

        :return: New InferRequest object.
        :rtype: openvino.runtime.InferRequest
        """
        return InferRequest(super().create_infer_request())

    def infer_new_request(self, inputs: Union[dict, list] = None) -> dict:
        """Infers specified input(s) in synchronous mode.

        Blocks all methods of CompiledModel while request is running.

        Method creates new temporary InferRequest and run inference on it.
        It is advised to use a dedicated InferRequest class for performance,
        optimizing workflows, and creating advanced pipelines.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.runtime.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.array`
        (2) `openvino.runtime.Tensor`

        :param inputs: Data to be set on input tensors.
        :type inputs: Union[Dict[keys, values], List[values]], optional
        :return: Dictionary of results from output tensors with ports as keys.
        :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        """
        return super().infer_new_request(
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )

    def __call__(self, inputs: Union[dict, list] = None) -> dict:
        """Callable infer wrapper for CompiledModel.

        Take a look at `infer_new_request` for reference.
        """
        return self.infer_new_request(inputs)


class AsyncInferQueue(AsyncInferQueueBase):
    """AsyncInferQueue with pool of asynchronous requests.

    AsyncInferQueue represents helper that creates a pool of asynchronous
    InferRequests and provides synchronization functions to control flow of
    a simple pipeline.
    """

    def __getitem__(self, i: int) -> InferRequest:
        """Gets InferRequest from the pool with given i id.

        :param i:  InferRequest id.
        :type i: int
        :return: InferRequests from the pool with given id.
        :rtype: openvino.runtime.InferRequest
        """
        return InferRequest(super().__getitem__(i))

    def start_async(
        self, inputs: Union[dict, list] = None, userdata: Any = None
    ) -> None:
        """Run asynchronous inference using the next available InferRequest from the pool.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.runtime.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.array`
        (2) `openvino.runtime.Tensor`

        :param inputs: Data to be set on input tensors of the next available InferRequest.
        :type inputs: Union[Dict[keys, values], List[values]], optional
        :param userdata: Any data that will be passed to a callback.
        :type userdata: Any, optional
        """
        super().start_async(
            {}
            if inputs is None
            else normalize_inputs(
                inputs, get_input_types(self[self.get_idle_request_id()])
            ),
            userdata,
        )


class Core(CoreBase):
    """Core class represents OpenVINO runtime Core entity.

    User applications can create several Core class instances, but in this
    case, the underlying plugins are created multiple times and not shared
    between several Core instances. The recommended way is to have a single
    Core instance per application.
    """

    def compile_model(
        self, model: Union[Model, str], device_name: str = None, config: dict = None
    ) -> CompiledModel:
        """Creates a compiled model.

        Creates a compiled model from a source Model object or
        reads model and creates a compiled model from IR / ONNX / PDPD file.
        This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
        especially for cases when caching is enabled and cached model is available.
        If device_name is not specified, the default OpenVINO device will be selected by AUTO plugin.
        Users can create as many compiled models as they need, and use them simultaneously
        (up to the limitation of the hardware resources).

        :param model: Model acquired from read_model function or a path to a model in IR / ONNX / PDPD format.
        :type model: Union[openvino.runtime.Model, str]
        :param device_name: Optional. Name of the device to load the model to. If not specified,
                            the default OpenVINO device will be selected by AUTO plugin.
        :type device_name: str
        :param config: Optional dict of pairs:
                       (property name, property value) relevant only for this load operation.
        :type config: dict
        :return: A compiled model.
        :rtype: openvino.runtime.CompiledModel
        """
        if device_name is None:
            return CompiledModel(
                super().compile_model(model, {} if config is None else config)
            )

        return CompiledModel(
            super().compile_model(model, device_name, {} if config is None else config)
        )

    def import_model(
        self, model_stream: bytes, device_name: str, config: dict = None
    ) -> CompiledModel:
        """Imports a compiled model from a previously exported one.

        :param model_stream: Input stream, containing a model previously exported, using export_model method.
        :type model_stream: bytes
        :param device_name: Name of device to which compiled model is imported.
                            Note: if device_name is not used to compile the original model,
                            an exception is thrown.
        :type device_name: str
        :param properties: Optional map of pairs: (property name,
                           property value) relevant only for this load operation.
        :type properties: dict, optional
        :return: A compiled model.
        :rtype: openvino.runtime.CompiledModel

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
        return CompiledModel(
            super().import_model(
                model_stream, device_name, {} if config is None else config
            )
        )


class ExtendedModel(CompiledModel):
    """CompiledModel that additionally holds Core object."""

    def __init__(self, core: Core, net: CompiledModel):
        super().__init__(net)
        self.core = core  # needs to store Core object for CPU plugin


def compile_model(model_path: str) -> CompiledModel:
    """Compact method to compile model with AUTO plugin.

    :param model_path: Path to file with model.
    :type model_path: str
    :return: Extended version of `CompiledModel` that holds and keeps alive `Core` object.
    """
    core = Core()
    return ExtendedModel(core, core.compile_model(model_path, "AUTO"))


class OVAny(OVAnyBase):
    """OVAny wrapper.

    Wrapper provides some useful overloads for simple built-in Python types.

    Access to the `OVAny` value is direct if it is a built-in Python data type.

    :Example:
    .. code-block:: ipython

        any = OVAny([1, 2])
        print(any[0])

        Output: 2

    Otherwise if `OVAny` value is a custom data type (for example user class),
    access to the value is possible by `get()` method or property 'value'.

    :Example:
    .. code-block:: python

        class Test:
            def __init__(self):
                self.data = "test"

        any = OVAny(Test())
        print(any.value.data)
    """

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self.value[key]

    def __get__(self) -> Any:
        return self.value

    def __setitem__(self, key: Union[str, int], val: Any) -> None:
        self.value[key] = val

    def __set__(self, val: Any) -> None:
        self.value = val

    def __len__(self) -> int:
        return len(self.value)
