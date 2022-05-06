// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/infer_request.hpp"

#include <ie_common.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/containers.hpp"

PYBIND11_MAKE_OPAQUE(Containers::TensorIndexMap);
PYBIND11_MAKE_OPAQUE(Containers::TensorNameMap);

namespace py = pybind11;

py::dict run_sync_infer(InferRequestWrapper& self) {
    {
        py::gil_scoped_release release;
        self._start_time = Time::now();
        self._request.infer();
        self._end_time = Time::now();
    }
    return Common::outputs_to_dict(self._outputs, self._request);
}

void regclass_InferRequest(py::module m) {
    py::class_<InferRequestWrapper, std::shared_ptr<InferRequestWrapper>> cls(m, "InferRequest");
    cls.doc() = "openvino.runtime.InferRequest represents infer request which can be run in asynchronous or "
                "synchronous manners.";

    cls.def(py::init([](InferRequestWrapper& other) {
                return other;
            }),
            py::arg("other"));

    // Python API exclusive function
    cls.def(
        "set_tensors",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            Common::set_request_tensors(self._request, inputs);
        },
        py::arg("inputs"),
        R"(
            Set tensors using given keys.

            :param inputs: Data to set on tensors.
            :type inputs: Dict[Union[int, str, openvino.runtime.ConstOutput], openvino.runtime.Tensor]
        )");

    cls.def(
        "set_tensors",
        [](InferRequestWrapper& self, const std::string& tensor_name, const std::vector<ov::Tensor>& tensors) {
            self._request.set_tensors(tensor_name, tensors);
        },
        py::arg("tensor_name"),
        py::arg("tensors"),
        R"(
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
            :type tensors: List[openvino.runtime.Tensor]
        )");

    cls.def(
        "set_tensors",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) {
            self._request.set_tensors(port, tensors);
        },
        py::arg("port"),
        py::arg("tensors"),
        R"(
            Sets batch of tensors for input data to infer by tensor name.
            Model input needs to have batch dimension and the number of tensors needs to be
            matched with batch size. Current version supports set tensors to model inputs only.
            In case if `port` is associated with output (or any other non-input node),
            an exception will be thrown.


            :param port: Port of input tensor.
            :type port: openvino.runtime.ConstOutput
            :param tensors: Input tensors for batched infer request. The type of each tensor
                            must match the model input element type and shape (except batch dimension).
                            Total size of tensors needs to match with input's size.
            :type tensors: List[openvino.runtime.Tensor]
            :rtype: None
        )");

    // Python API exclusive function
    cls.def(
        "set_output_tensors",
        [](InferRequestWrapper& self, const py::dict& outputs) {
            auto outputs_map = Common::cast_to_tensor_index_map(outputs);
            for (auto&& output : outputs_map) {
                self._request.set_output_tensor(output.first, output.second);
            }
        },
        py::arg("outputs"),
        R"(
            Set output tensors using given indexes.

            :param inputs: Data to set on output tensors.
            :type inputs: Dict[int, openvino.runtime.Tensor]
        )");

    // Python API exclusive function
    cls.def(
        "set_input_tensors",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            auto inputs_map = Common::cast_to_tensor_index_map(inputs);
            for (auto&& input : inputs_map) {
                self._request.set_input_tensor(input.first, input.second);
            }
        },
        py::arg("inputs"),
        R"(
            Set input tensors using given indexes.

            :param inputs: Data to set on output tensors.
            :type inputs: Dict[int, openvino.runtime.Tensor]
        )");

    cls.def(
        "set_input_tensors",
        [](InferRequestWrapper& self, const std::vector<ov::Tensor>& tensors) {
            self._request.set_input_tensors(tensors);
        },
        py::arg("tensors"),
        R"(
            Sets batch of tensors for single input data.
            Model input needs to have batch dimension and the number of `tensors`
            needs to match with batch size.

            :param tensors:  Input tensors for batched infer request. The type of each tensor
                             must match the model input element type and shape (except batch dimension).
                             Total size of tensors needs to match with input's size.
            :type tensors: List[openvino.runtime.Tensor]
        )");

    cls.def(
        "set_input_tensors",
        [](InferRequestWrapper& self, size_t idx, const std::vector<ov::Tensor>& tensors) {
            self._request.set_input_tensors(idx, tensors);
        },
        py::arg("idx"),
        py::arg("tensors"),
        R"(
            Sets batch of tensors for single input data to infer by index.
            Model input needs to have batch dimension and the number of `tensors`
            needs to match with batch size.

            :param idx: Index of input tensor.
            :type idx: int
            :param tensors: Input tensors for batched infer request. The type of each tensor
                            must match the model input element type and shape (except batch dimension).
                            Total size of tensors needs to match with input's size.
        )");

    // Overload for single input, it will throw error if a model has more than one input.
    cls.def(
        "infer",
        [](InferRequestWrapper& self, const ov::Tensor& inputs) {
            self._request.set_input_tensor(inputs);
            return run_sync_infer(self);
        },
        py::arg("inputs"),
        R"(
            Infers specified input(s) in synchronous mode.
            Blocks all methods of InferRequest while request is running.
            Calling any method will lead to throwing exceptions.

            GIL is released while running the inference.

            :param inputs: Data to set on single input tensor.
            :type inputs: openvino.runtime.Tensor
            :return: Dictionary of results from output tensors with ports as keys.
            :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        )");

    // Overload for general case, it accepts dict of inputs that are pairs of (key, value).
    // Where keys types are:
    // * ov::Output<const ov::Node>
    // * py::str (std::string)
    // * py::int_ (size_t)
    // and values are always of type: ov::Tensor.
    cls.def(
        "infer",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            // Update inputs if there are any
            Common::set_request_tensors(self._request, inputs);
            // Call Infer function
            return run_sync_infer(self);
        },
        py::arg("inputs"),
        R"(
            Infers specified input(s) in synchronous mode.
            Blocks all methods of InferRequest while request is running.
            Calling any method will lead to throwing exceptions.

            GIL is released while running the inference.

            :param inputs: Data to set on input tensors.
            :type inputs: Dict[Union[int, str, openvino.runtime.ConstOutput], openvino.runtime.Tensor]
            :return: Dictionary of results from output tensors with ports as keys.
            :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        )");

    // Overload for single input, it will throw error if a model has more than one input.
    cls.def(
        "start_async",
        [](InferRequestWrapper& self, const ov::Tensor& inputs, py::object& userdata) {
            // Update inputs if there are any
            self._request.set_input_tensor(inputs);
            if (!userdata.is(py::none())) {
                if (self.user_callback_defined) {
                    self.userdata = userdata;
                } else {
                    PyErr_WarnEx(PyExc_RuntimeWarning, "There is no callback function!", 1);
                }
            }
            py::gil_scoped_release release;
            self._start_time = Time::now();
            self._request.start_async();
        },
        py::arg("inputs"),
        py::arg("userdata"),
        R"(
            Starts inference of specified input(s) in asynchronous mode.
            Returns immediately. Inference starts also immediately.

            GIL is released while running the inference.

            Calling any method on this InferRequest while the request is
            running will lead to throwing exceptions.

            :param inputs: Data to set on single input tensors.
            :type inputs: openvino.runtime.Tensor
            :param userdata: Any data that will be passed inside callback call.
            :type userdata: Any
        )");

    // Overload for general case, it accepts dict of inputs that are pairs of (key, value).
    // Where keys types are:
    // * ov::Output<const ov::Node>
    // * py::str (std::string)
    // * py::int_ (size_t)
    // and values are always of type: ov::Tensor.
    cls.def(
        "start_async",
        [](InferRequestWrapper& self, const py::dict& inputs, py::object& userdata) {
            // Update inputs if there are any
            Common::set_request_tensors(self._request, inputs);
            if (!userdata.is(py::none())) {
                if (self.user_callback_defined) {
                    self.userdata = userdata;
                } else {
                    PyErr_WarnEx(PyExc_RuntimeWarning, "There is no callback function!", 1);
                }
            }
            py::gil_scoped_release release;
            self._start_time = Time::now();
            self._request.start_async();
        },
        py::arg("inputs"),
        py::arg("userdata"),
        R"(
            Starts inference of specified input(s) in asynchronous mode.
            Returns immediately. Inference starts also immediately.

            GIL is released while running the inference.

            Calling any method on this InferRequest while the request is
            running will lead to throwing exceptions.

            :param inputs: Data to set on input tensors.
            :type inputs: Dict[Union[int, str, openvino.runtime.ConstOutput], openvino.runtime.Tensor]
            :param userdata: Any data that will be passed inside callback call.
            :type userdata: Any
        )");

    cls.def(
        "cancel",
        [](InferRequestWrapper& self) {
            self._request.cancel();
        },
        R"(
            Cancels inference request.
        )");

    cls.def(
        "wait",
        [](InferRequestWrapper& self) {
            py::gil_scoped_release release;
            self._request.wait();
        },
        R"(
            Waits for the result to become available. 
            Blocks until the result becomes available.

            GIL is released while running this function.
        )");

    cls.def(
        "wait_for",
        [](InferRequestWrapper& self, const int timeout) {
            py::gil_scoped_release release;
            return self._request.wait_for(std::chrono::milliseconds(timeout));
        },
        py::arg("timeout"),
        R"(
            Waits for the result to become available.
            Blocks until specified timeout has elapsed or
            the result becomes available, whichever comes first.

            GIL is released while running this function.

            :param timeout: Maximum duration in milliseconds (ms) of blocking call.
            :type timeout: int
            :return: True if InferRequest is ready, False otherwise.
            :rtype: bool
        )");

    cls.def(
        "set_callback",
        [](InferRequestWrapper& self, py::function callback, py::object& userdata) {
            self.userdata = userdata;
            self.user_callback_defined = true;
            self._request.set_callback([&self, callback](std::exception_ptr exception_ptr) {
                self._end_time = Time::now();
                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                } catch (const std::exception& e) {
                    throw ov::Exception("Caught exception: " + std::string(e.what()));
                }
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                callback(self.userdata);
            });
        },
        py::arg("callback"),
        py::arg("userdata"),
        R"(
            Sets a callback function that will be called on success or failure of asynchronous InferRequest.

            :param callback: Function defined in Python.
            :type callback: function
            :param userdata: Any data that will be passed inside callback call.
            :type userdata: Any
        )");

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const std::string& name) {
            return self._request.get_tensor(name);
        },
        py::arg("name"),
        R"(
            Gets input/output tensor of InferRequest.

            :param name: Name of tensor to get.
            :type name: str
            :return: A Tensor object with given name.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"),
        R"(
            Gets input/output tensor of InferRequest.

            :param port: Port of tensor to get.
            :type port: openvino.runtime.ConstOutput
            :return: A Tensor object for the port.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"),
        R"(
            Gets input/output tensor of InferRequest.

            :param port: Port of tensor to get.
            :type port: openvino.runtime.Output
            :return: A Tensor object for the port.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "get_input_tensor",
        [](InferRequestWrapper& self, size_t idx) {
            return self._request.get_input_tensor(idx);
        },
        py::arg("index"),
        R"(
            Gets input tensor of InferRequest.

            :param idx: An index of tensor to get.
            :type idx: int
            :return: An input Tensor with index idx for the model.
                     If a tensor with specified idx is not found,
            an exception is thrown.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "get_input_tensor",
        [](InferRequestWrapper& self) {
            return self._request.get_input_tensor();
        },
        R"(
            Gets input tensor of InferRequest.

            :return: An input Tensor for the model.
                     If model has several inputs, an exception is thrown.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "get_output_tensor",
        [](InferRequestWrapper& self, size_t idx) {
            return self._request.get_output_tensor(idx);
        },
        py::arg("index"),
        R"(
            Gets output tensor of InferRequest.

            :param idx: An index of tensor to get.
            :type idx: int
            :return: An output Tensor with index idx for the model.
                     If a tensor with specified idx is not found, an exception is thrown.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "get_output_tensor",
        [](InferRequestWrapper& self) {
            return self._request.get_output_tensor();
        },
        R"(
            Gets output tensor of InferRequest.
            
            :return: An output Tensor for the model.
                     If model has several outputs, an exception is thrown.
            :rtype: openvino.runtime.Tensor
        )");

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const std::string& name, const ov::Tensor& tensor) {
            self._request.set_tensor(name, tensor);
        },
        py::arg("name"),
        py::arg("tensor"),
        R"(
            Sets input/output tensor of InferRequest.

            :param name: Name of input/output tensor.
            :type name: str
            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's input/output element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
            self._request.set_tensor(port, tensor);
        },
        py::arg("port"),
        py::arg("tensor"),
        R"(
            Sets input/output tensor of InferRequest.

            :param port: Port of input/output tensor.
            :type port: openvino.runtime.ConstOutput
            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's input/output element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port, const ov::Tensor& tensor) {
            self._request.set_tensor(port, tensor);
        },
        py::arg("port"),
        py::arg("tensor"),
        R"(
            Sets input/output tensor of InferRequest.

            :param port: Port of input/output tensor.
            :type port: openvino.runtime.Output
            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's input/output element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "set_input_tensor",
        [](InferRequestWrapper& self, size_t idx, const ov::Tensor& tensor) {
            self._request.set_input_tensor(idx, tensor);
        },
        py::arg("index"),
        py::arg("tensor"),
        R"(
            Sets input tensor of InferRequest.

            :param idx: Index of input tensor. If idx is greater than number of model's inputs,
                        an exception is thrown.
            :type idx: int
            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's input element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "set_input_tensor",
        [](InferRequestWrapper& self, const ov::Tensor& tensor) {
            self._request.set_input_tensor(tensor);
        },
        py::arg("tensor"),
        R"(
            Sets input tensor of InferRequest with single input.
            If model has several inputs, an exception is thrown.

            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's input element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "set_output_tensor",
        [](InferRequestWrapper& self, size_t idx, const ov::Tensor& tensor) {
            self._request.set_output_tensor(idx, tensor);
        },
        py::arg("index"),
        py::arg("tensor"),
        R"(
            Sets output tensor of InferRequest.

            :param idx: Index of output tensor.
            :type idx: int
            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's output element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "set_output_tensor",
        [](InferRequestWrapper& self, const ov::Tensor& tensor) {
            self._request.set_output_tensor(tensor);
        },
        py::arg("tensor"),
        R"(
            Sets output tensor of InferRequest with single output.
            If model has several outputs, an exception is thrown.

            :param tensor: Tensor object. The element_type and shape of a tensor
                           must match the model's output element_type and shape.
            :type tensor: openvino.runtime.Tensor
        )");

    cls.def(
        "get_profiling_info",
        [](InferRequestWrapper& self) {
            return self._request.get_profiling_info();
        },
        py::call_guard<py::gil_scoped_release>(),
        R"(
            Queries performance is measured per layer to get feedback on what
            is the most time-consuming operation, not all plugins provide
            meaningful data.

            GIL is released while running this function.

            :return: List of profiling information for operations in model.
            :rtype: List[openvino.runtime.ProfilingInfo]
        )");

    cls.def(
        "query_state",
        [](InferRequestWrapper& self) {
            return self._request.query_state();
        },
        py::call_guard<py::gil_scoped_release>(),
        R"(
            Gets state control interface for given infer request.

            GIL is released while running this function.

            :return: List of VariableState objects.
            :rtype: List[openvino.runtime.VariableState]
        )");

    cls.def_property_readonly(
        "userdata",
        [](InferRequestWrapper& self) {
            return self.userdata;
        },
        R"(
            Gets currently held userdata.

            :rtype: Any
        )");

    cls.def_property_readonly(
        "model_inputs",
        [](InferRequestWrapper& self) {
            return self._inputs;
        },
        R"(
            Gets all inputs of a compiled model which was used to create this InferRequest.

            :rtype: List[openvino.runtime.ConstOutput]
        )");

    cls.def_property_readonly(
        "model_outputs",
        [](InferRequestWrapper& self) {
            return self._outputs;
        },
        R"(
            Gets all outputs of a compiled model which was used to create this InferRequest.

            :rtype: List[openvino.runtime.ConstOutput]
        )");

    cls.def_property_readonly("inputs",
                              &InferRequestWrapper::get_input_tensors,
                              R"(
                                Gets all input tensors of this InferRequest.
                                
                                :rtype: List[openvino.runtime.Tensor]
                              )");

    cls.def_property_readonly("outputs",
                              &InferRequestWrapper::get_output_tensors,
                              R"(
                                Gets all output tensors of this InferRequest.
                                
                                :rtype: List[openvino.runtime.Tensor]
                              )");

    cls.def_property_readonly("input_tensors",
                              &InferRequestWrapper::get_input_tensors,
                              R"(
                                Gets all input tensors of this InferRequest.
                                
                                :rtype: List[openvino.runtime.Tensor]
                              )");

    cls.def_property_readonly("output_tensors",
                              &InferRequestWrapper::get_output_tensors,
                              R"(

                                Gets all output tensors of this InferRequest.
                                
                                :rtype: List[openvino.runtime.Tensor]
                              )");

    cls.def_property_readonly(
        "latency",
        [](InferRequestWrapper& self) {
            return self.get_latency();
        },
        R"(
            Gets latency of this InferRequest.
            
            :rtype: float
        )");

    cls.def_property_readonly(
        "profiling_info",
        [](InferRequestWrapper& self) {
            return self._request.get_profiling_info();
        },
        py::call_guard<py::gil_scoped_release>(),
        R"(
            Performance is measured per layer to get feedback on the most time-consuming operation.
            Not all plugins provide meaningful data!

            GIL is released while running this function.
            
            :return: Inference time.
            :rtype: List[openvino.runtime.ProfilingInfo]
        )");

    cls.def_property_readonly(
        "results",
        [](InferRequestWrapper& self) {
            return Common::outputs_to_dict(self._outputs, self._request);
        },
        R"(
            Gets all outputs tensors of this InferRequest.

            :return: Dictionary of results from output tensors with ports as keys.
            :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        )");

    cls.def("__repr__", [](const InferRequestWrapper& self) {
        auto inputs_str = Common::docs::container_to_string(self._inputs, ",\n");
        auto outputs_str = Common::docs::container_to_string(self._outputs, ",\n");

        return "<InferRequest:\ninputs[\n" + inputs_str + "\n]\noutputs[\n" + outputs_str + "\n]>";
    });
}
