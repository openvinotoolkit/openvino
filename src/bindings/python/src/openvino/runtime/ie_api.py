# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch
from typing import Any, Union, Dict
from pathlib import Path

import numpy as np

from openvino.pyopenvino import Model
from openvino.pyopenvino import Core as CoreBase
from openvino.pyopenvino import CompiledModel as CompiledModelBase
from openvino.pyopenvino import InferRequest as InferRequestBase
from openvino.pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino.pyopenvino import ConstOutput
from openvino.pyopenvino import Tensor


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))  # type: ignore


def set_scalar_tensor(request: InferRequestBase, tensor: Tensor, key: Union[str, int, ConstOutput] = None) -> None:
    if key is None:
        request.set_input_tensor(tensor)
    elif isinstance(key, int):
        request.set_input_tensor(key, tensor)
    elif isinstance(key, (str, ConstOutput)):
        request.set_tensor(key, tensor)
    else:
        raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")


@singledispatch
def update_tensor(
    inputs: Union[np.ndarray, np.number, int, float],
    request: InferRequestBase,
    key: Union[str, int, ConstOutput] = None,
) -> None:
    raise TypeError(f"Incompatible input data of type {type(inputs)} under {key} key!")


@update_tensor.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: InferRequestBase,
    key: Union[str, int, ConstOutput] = None,
) -> None:
    # If shape is "empty", assume this is a scalar value
    if not inputs.shape:
        set_scalar_tensor(request, Tensor(inputs), key)
    else:
        if key is None:
            tensor = request.get_input_tensor()
        elif isinstance(key, int):
            tensor = request.get_input_tensor(key)
        elif isinstance(key, (str, ConstOutput)):
            tensor = request.get_tensor(key)
        else:
            raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")
        # Update shape if there is a mismatch
        if tensor.shape != inputs.shape:
            tensor.shape = inputs.shape
        # When copying, type should be up/down-casted automatically.
        tensor.data[:] = inputs[:]


@update_tensor.register(np.number)
@update_tensor.register(float)
@update_tensor.register(int)
def _(
    inputs: Union[np.number, float, int],
    request: InferRequestBase,
    key: Union[str, int, ConstOutput] = None,
) -> None:
    set_scalar_tensor(
        request, Tensor(np.ndarray([], type(inputs), np.array(inputs))), key,
    )


def normalize_inputs(request: InferRequestBase, inputs: dict) -> dict:
    """Helper function to prepare inputs for inference.

    It creates copy of Tensors or copy data to already allocated Tensors on device
    if the item is of type `np.ndarray`, `np.number`, `int`, `float` or has numpy __array__ attribute.
    """
    # Create new temporary dictionary.
    # new_inputs will be used to transfer data to inference calls,
    # ensuring that original inputs are not overwritten with Tensors.
    new_inputs: Dict[Union[str, int, ConstOutput], Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(key, (str, int, ConstOutput)):
            raise TypeError(f"Incompatible key type for input: {key}")
        # Copy numpy arrays to already allocated Tensors.
        if isinstance(value, (np.ndarray, np.number, int, float)):
            update_tensor(value, request, key)
        # If value is of Tensor type, put it into temporary dictionary.
        elif isinstance(value, Tensor):
            new_inputs[key] = value
        # If value object has __array__ attribute, load it to Tensor using np.array.
        elif hasattr(value, "__array__"):
            update_tensor(np.array(value, copy=True), request, key)
        # Throw error otherwise.
        else:
            raise TypeError(f"Incompatible input data of type {type(value)} under {key} key!")
    return new_inputs


class InferRequest(InferRequestBase):
    """InferRequest class represents infer request which can be run in asynchronous or synchronous manners."""

    def infer(self, inputs: Any = None) -> dict:
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
        (3) array-like object with `__array__` attribute

        Can be called with only one `openvino.runtime.Tensor` or `numpy.array`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Any, optional
        :return: Dictionary of results from output tensors with ports as keys.
        :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        """
        # If inputs are empty, pass empty dictionary.
        if inputs is None:
            return super().infer({})
        # If inputs are dict, normalize dictionary and call infer method.
        elif isinstance(inputs, dict):
            return super().infer(normalize_inputs(self, inputs))
        # If inputs are list or tuple, enumarate inputs and save them as dictionary.
        # It is an extension of above branch with dict inputs.
        elif isinstance(inputs, (list, tuple)):
            return super().infer(
                normalize_inputs(self, {index: input for index, input in enumerate(inputs)}))
        # If inputs are Tensor, call infer method directly.
        elif isinstance(inputs, Tensor):
            return super().infer(inputs)
        # If inputs are single numpy array or scalars, use helper function to copy them
        # directly to Tensor or create temporary Tensor to pass into the InferRequest.
        # Pass empty dictionary to infer method, inputs are already set by helper function.
        elif isinstance(inputs, (np.ndarray, np.number, int, float)):
            update_tensor(inputs, self)
            return super().infer({})
        elif hasattr(inputs, "__array__"):
            update_tensor(np.array(inputs, copy=True), self)
            return super().infer({})
        else:
            raise TypeError(f"Incompatible inputs of type: {type(inputs)}")

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
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
        (3) array-like object with `__array__` attribute

        Can be called with only one `openvino.runtime.Tensor` or `numpy.array`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Any, optional
        :param userdata: Any data that will be passed inside the callback.
        :type userdata: Any
        """
        if inputs is None:
            super().start_async({}, userdata)
        elif isinstance(inputs, dict):
            super().start_async(normalize_inputs(self, inputs), userdata)
        elif isinstance(inputs, (list, tuple)):
            super().start_async(
                normalize_inputs(self, {index: input for index, input in enumerate(inputs)}), userdata)
        elif isinstance(inputs, Tensor):
            super().start_async(inputs, userdata)
        elif isinstance(inputs, (np.ndarray, np.number, int, float)):
            update_tensor(inputs, self)
            return super().start_async({}, userdata)
        elif hasattr(inputs, "__array__"):
            update_tensor(np.array(inputs, copy=True), self)
            return super().start_async({}, userdata)
        else:
            raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


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

    def infer_new_request(self, inputs: Union[dict, list, tuple, Tensor, np.ndarray] = None) -> dict:
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

        Can be called with only one `openvino.runtime.Tensor` or `numpy.array`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Union[Dict[keys, values], List[values], Tuple[values], Tensor, numpy.array], optional
        :return: Dictionary of results from output tensors with ports as keys.
        :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        """
        # It returns wrapped python InferReqeust and then call upon
        # overloaded functions of InferRequest class
        return self.create_infer_request().infer(inputs)

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
        self,
        inputs: Any = None,
        userdata: Any = None,
    ) -> None:
        """Run asynchronous inference using the next available InferRequest from the pool.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.runtime.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.array`
        (2) `openvino.runtime.Tensor`
        (3) array-like object with `__array__` attribute

        Can be called with only one `openvino.runtime.Tensor` or `numpy.array`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors of the next available InferRequest.
        :type inputs: Any, optional
        :param userdata: Any data that will be passed to a callback.
        :type userdata: Any, optional
        """
        if inputs is None:
            super().start_async({}, userdata)
        elif isinstance(inputs, dict):
            super().start_async(
                normalize_inputs(self[self.get_idle_request_id()], inputs), userdata,
            )
        elif isinstance(inputs, (list, tuple)):
            super().start_async(
                normalize_inputs(
                    self[self.get_idle_request_id()],
                    {index: input for index, input in enumerate(inputs)},
                ),
                userdata,
            )
        elif isinstance(inputs, Tensor):
            super().start_async(inputs, userdata)
        elif isinstance(inputs, (np.ndarray, np.number, int, float)):
            update_tensor(inputs, self[self.get_idle_request_id()])
            super().start_async({}, userdata)
        elif hasattr(inputs, "__array__"):
            update_tensor(np.array(inputs, copy=True), self[self.get_idle_request_id()])
            super().start_async({}, userdata)
        else:
            raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


class Core(CoreBase):
    """Core class represents OpenVINO runtime Core entity.

    User applications can create several Core class instances, but in this
    case, the underlying plugins are created multiple times and not shared
    between several Core instances. The recommended way is to have a single
    Core instance per application.
    """

    def compile_model(
        self, model: Union[Model, str, Path], device_name: str = None, config: dict = None,
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
        :type model: Union[openvino.runtime.Model, str, pathlib.Path]
        :param device_name: Optional. Name of the device to load the model to. If not specified,
                            the default OpenVINO device will be selected by AUTO plugin.
        :type device_name: str
        :param config: Optional dict of pairs:
                       (property name, property value) relevant only for this load operation.
        :type config: dict, optional
        :return: A compiled model.
        :rtype: openvino.runtime.CompiledModel
        """
        if device_name is None:
            return CompiledModel(
                super().compile_model(model, {} if config is None else config),
            )

        return CompiledModel(
            super().compile_model(model, device_name, {} if config is None else config),
        )

    def import_model(
        self, model_stream: bytes, device_name: str, config: dict = None,
    ) -> CompiledModel:
        """Imports a compiled model from a previously exported one.

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
                model_stream, device_name, {} if config is None else config,
            ),
        )


def compile_model(model_path: Union[str, Path]) -> CompiledModel:
    """Compact method to compile model with AUTO plugin.

    :param model_path: Path to file with model.
    :type model_path: str, pathlib.Path
    :return: A compiled model
    :rtype: openvino.runtime.CompiledModel

    """
    core = Core()
    return core.compile_model(model_path, "AUTO")
