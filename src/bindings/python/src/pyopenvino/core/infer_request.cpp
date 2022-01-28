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

void regclass_InferRequest(py::module m) {
    py::class_<InferRequestWrapper, std::shared_ptr<InferRequestWrapper>> cls(m, "InferRequest");
    cls.doc() = "openvino.runtime.InferRequest represents infer request which
                 can be run in asynchronous or synchronous manners.";

    cls.def(py::init([](InferRequestWrapper& other) {
                return other;
            }),
            py::arg("other"));

    cls.def(
        "set_tensors",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            auto tensor_map = Common::cast_to_tensor_name_map(inputs);
            for (auto&& input : tensor_map) {
                self._request.set_tensor(input.first, input.second);
            }
        },
        py::arg("inputs"));

    cls.def(
        "set_output_tensors",
        [](InferRequestWrapper& self, const py::dict& outputs) {
            auto outputs_map = Common::cast_to_tensor_index_map(outputs);
            for (auto&& output : outputs_map) {
                self._request.set_output_tensor(output.first, output.second);
            }
        },
        py::arg("outputs"));

    cls.def(
        "set_input_tensors",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            auto inputs_map = Common::cast_to_tensor_index_map(inputs);
            for (auto&& input : inputs_map) {
                self._request.set_input_tensor(input.first, input.second);
            }
        },
        py::arg("inputs"));

    cls.def(
        "infer",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            // Update inputs if there are any
            Common::set_request_tensors(self._request, inputs);
            // Call Infer function
            self._start_time = Time::now();
            self._request.infer();
            self._end_time = Time::now();
            return Common::outputs_to_dict(self._outputs, self._request);
        },
        py::arg("inputs"),
        R"(
            Infers specified input(s) in synchronous mode.
            Blocks all methods of InferRequest while request is running.
            Calling any method will lead to throwning exceptions.

            Parameters
            ----------
            inputs : dict[Union[int, str, openvino.runtime.ConstOutput] : openvino.runtime.Tensor]
                Data to set on input tensors.

            Returns
            ----------
            infer : dict[openvino.runtime.ConstOutput : openvino.runtime.Tensor]
                Dictionary of results from output tensors with ports as keys.
        )");

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
            Calling any method while the request is running will lead to
            throwning exceptions.

            Parameters
            ----------
            inputs : dict[Union[int, str, openvino.runtime.ConstOutput] : openvino.runtime.Tensor]
                Data to set on input tensors.

            userdata : Any
                Any data that will be passed inside callback call.                

            Returns
            ----------
            start_async : None
        )");

    cls.def("cancel", [](InferRequestWrapper& self) {
        self._request.cancel();
    },
    R"(
        Cancels inference request.

        Parameters
        ----------
        None

        Returns
        ----------
        cancel : None
    )");

    cls.def("wait", [](InferRequestWrapper& self) {
        py::gil_scoped_release release;
        self._request.wait();
    },
    R"(
        Waits for the result to become available. 
        Blocks until the result becomes available.

        Parameters
        ----------
        None

        Returns
        ----------
        wait : None
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

            Parameters
            ----------
            timeout : int
                Maximum duration in milliseconds (ms) of blocking call.

            Returns
            ----------
            wait_for : bool
                True if InferRequest is ready, False otherwise.
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
            Sets a callback function that will be called on success or
            failure of asynchronous InferRequest.

            Parameters
            ----------
            callback : function
                Function defined in Python.

            userdata : Any
                Any data that will be passed inside callback call.

            Returns
            ----------
            set_callback : None // TAK TO ROBIĆ!!!
        )");

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const std::string& name) {
            return self._request.get_tensor(name);
        },
        py::arg("name"),
        R"(
            Gets input/output tensor of InferRequest.

            Parameters
            ----------
            name : str
                Name of tensor to get.

            Returns
            ----------
            get_tensor : openvino.runtime.Tensor
                A Tensor object with given name.
        )");

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"),
        R"(
            Gets input/output tensor of InferRequest.

            Parameters
            ----------
            port : openvino.runtime.ConstOutput
                Port of tensor to get.

            Returns
            ----------
            get_tensor : openvino.runtime.Tensor
                A Tensor object for the port.
        )");

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"),
        R"(
            Gets input/output tensor of InferRequest.

            Parameters
            ----------
            port : openvino.runtime.Output
                Port of tensor to get.

            Returns
            ----------
            get_tensor : openvino.runtime.Tensor
                A Tensor object for the port.
        )");

    cls.def(
        "get_input_tensor",
        [](InferRequestWrapper& self, size_t idx) {
            return self._request.get_input_tensor(idx);
        },
        py::arg("index"),
        R"(
            Gets input tensor of InferRequest.

            Parameters
            ----------
            idx : int
                An index of tensor to get.

            Returns
            ----------
            get_input_tensor : openvino.runtime.Tensor
                An input Tensor with index idx for the model.
                If a tensor with specified idx is not found,
                an exception is thrown.
        )");

    cls.def("get_input_tensor", [](InferRequestWrapper& self) {
        return self._request.get_input_tensor();
    },
    R"(
        Gets input tensor of InferRequest.

        Parameters
        ----------
        None

        Returns
        ----------
        get_input_tensor : openvino.runtime.Tensor
            An input Tensor for the model.
            If model has several inputs, an exception is thrown.
    )");

    cls.def(
        "get_output_tensor",
        [](InferRequestWrapper& self, size_t idx) {
            return self._request.get_output_tensor(idx);
        },
        py::arg("index"),
        R"(
            Gets output tensor of InferRequest.

            Parameters
            ----------
            idx : int
                An index of tensor to get.

            Returns
            ----------
            get_output_tensor : openvino.runtime.Tensor
                An output Tensor with index idx for the model.
                If a tensor with specified idx is not found,
                an exception is thrown.
        )");

    cls.def("get_output_tensor", [](InferRequestWrapper& self) {
        return self._request.get_output_tensor();
    },
    R"(
        Gets output tensor of InferRequest.

        Parameters
        ----------
        None

        Returns
        ----------
        get_output_tensor : openvino.runtime.Tensor
            An output Tensor for the model.
            If model has several outputs, an exception is thrown.
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

            Parameters
            ----------
            name : str
                Name of input/output tensor.

            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's input/output element_type and shape.

            Returns
            ----------
            set_tensor : None
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

            Parameters
            ----------
            port : openvino.runtime.ConstOutput
                Port of input/output tensor.

            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's input/output element_type and shape.

            Returns
            ----------
            set_tensor : None
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

            Parameters
            ----------
            port : openvino.runtime.Output
                Port of input/output tensor.

            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's input/output element_type and shape.

            Returns
            ----------
            set_tensor : None
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

            Parameters
            ----------
            idx : int
                Index of input tensor.
                If idx is greater than number of model's inputs,
                an exception is thrown.

            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's input element_type and shape.

            Returns
            ----------
            set_input_tensor : None
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

            Parameters
            ----------
            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's input element_type and shape.

            Returns
            ----------
            set_input_tensor : None
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

            Parameters
            ----------
            idx : int
                Index of output tensor.

            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's output element_type and shape.

            Returns
            ----------
            set_output_tensor : None
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

            Parameters
            ----------
            tensor : openvino.runtime.Tensor
                Tensor object. The element_type and shape of a tensor
                must match the model's output element_type and shape.

            Returns
            ----------
            set_output_tensor : None
        )");

    cls.def("get_profiling_info", [](InferRequestWrapper& self) {
        return self._request.get_profiling_info();
    },
    R"(
        Queries performance measures per layer to get feedback of what
        is the most time consuming operation, not all plugins provide
        meaningful data.

        Parameters
        ----------
        None

        Returns
        ----------
        get_profiling_info : list[openvino.runtime.ProfilingInfo]
            List of profiling information for operations in model.
    )");

    cls.def("query_state", [](InferRequestWrapper& self) {
        return self._request.query_state();
    },
    R"(
        Gets state control interface for given infer request.

        Parameters
        ----------
        None

        Returns
        ----------
        query_state : list[openvino.runtime.VariableState]
            List of VariableState objects.
    )");

    cls.def_property_readonly("userdata", [](InferRequestWrapper& self) {
        return self.userdata;
    });

    cls.def_property_readonly("model_inputs", [](InferRequestWrapper& self) {
        return self._inputs;
    });

    cls.def_property_readonly("model_outputs", [](InferRequestWrapper& self) {
        return self._outputs;
    });

    cls.def_property_readonly("inputs", &InferRequestWrapper::get_input_tensors);

    cls.def_property_readonly("outputs", &InferRequestWrapper::get_output_tensors);

    cls.def_property_readonly("input_tensors", &InferRequestWrapper::get_input_tensors);

    cls.def_property_readonly("output_tensors", &InferRequestWrapper::get_output_tensors);

    cls.def_property_readonly("latency", [](InferRequestWrapper& self) {
        return self.get_latency();
    });

    cls.def_property_readonly("profiling_info", [](InferRequestWrapper& self) {
        return self._request.get_profiling_info();
    });

    cls.def_property_readonly("results", [](InferRequestWrapper& self) {
        return Common::outputs_to_dict(self._outputs, self._request);
    });
}
