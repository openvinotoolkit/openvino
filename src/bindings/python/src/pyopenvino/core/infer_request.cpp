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
        py::arg("inputs"));

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
        py::arg("userdata"));

    cls.def("cancel", [](InferRequestWrapper& self) {
        self._request.cancel();
    });

    cls.def("wait", [](InferRequestWrapper& self) {
        py::gil_scoped_release release;
        self._request.wait();
    });

    cls.def(
        "wait_for",
        [](InferRequestWrapper& self, const int timeout) {
            py::gil_scoped_release release;
            return self._request.wait_for(std::chrono::milliseconds(timeout));
        },
        py::arg("timeout"));

    cls.def(
        "set_callback",
        [](InferRequestWrapper& self, py::function f_callback, py::object& userdata) {
            self.userdata = userdata;
            self.user_callback_defined = true;
            self._request.set_callback([&self, f_callback](std::exception_ptr exception_ptr) {
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
                f_callback(self.userdata);
            });
        },
        py::arg("f_callback"),
        py::arg("userdata"));

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const std::string& name) {
            return self._request.get_tensor(name);
        },
        py::arg("name"));

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"));

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"));

    cls.def(
        "get_input_tensor",
        [](InferRequestWrapper& self, size_t idx) {
            return self._request.get_input_tensor(idx);
        },
        py::arg("index"));

    cls.def("get_input_tensor", [](InferRequestWrapper& self) {
        return self._request.get_input_tensor();
    });

    cls.def(
        "get_output_tensor",
        [](InferRequestWrapper& self, size_t idx) {
            return self._request.get_output_tensor(idx);
        },
        py::arg("index"));

    cls.def("get_output_tensor", [](InferRequestWrapper& self) {
        return self._request.get_output_tensor();
    });

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const std::string& name, const ov::runtime::Tensor& tensor) {
            self._request.set_tensor(name, tensor);
        },
        py::arg("name"),
        py::arg("tensor"));

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port, const ov::runtime::Tensor& tensor) {
            self._request.set_tensor(port, tensor);
        },
        py::arg("port"),
        py::arg("tensor"));

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port, const ov::runtime::Tensor& tensor) {
            self._request.set_tensor(port, tensor);
        },
        py::arg("port"),
        py::arg("tensor"));

    cls.def(
        "set_input_tensor",
        [](InferRequestWrapper& self, size_t idx, const ov::runtime::Tensor& tensor) {
            self._request.set_input_tensor(idx, tensor);
        },
        py::arg("index"),
        py::arg("tensor"));

    cls.def(
        "set_input_tensor",
        [](InferRequestWrapper& self, const ov::runtime::Tensor& tensor) {
            self._request.set_input_tensor(tensor);
        },
        py::arg("tensor"));

    cls.def(
        "set_output_tensor",
        [](InferRequestWrapper& self, size_t idx, const ov::runtime::Tensor& tensor) {
            self._request.set_output_tensor(idx, tensor);
        },
        py::arg("index"),
        py::arg("tensor"));

    cls.def(
        "set_output_tensor",
        [](InferRequestWrapper& self, const ov::runtime::Tensor& tensor) {
            self._request.set_output_tensor(tensor);
        },
        py::arg("tensor"));

    cls.def("get_profiling_info", [](InferRequestWrapper& self) {
        return self._request.get_profiling_info();
    });

    cls.def("query_state", [](InferRequestWrapper& self) {
        return self._request.query_state();
    });

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
