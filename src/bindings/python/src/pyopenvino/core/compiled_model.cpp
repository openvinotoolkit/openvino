// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include <pybind11/stl.h>

#include "common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/infer_request.hpp"

PYBIND11_MAKE_OPAQUE(Containers::TensorIndexMap);
PYBIND11_MAKE_OPAQUE(Containers::TensorNameMap);

namespace py = pybind11;

void regclass_CompiledModel(py::module m) {
    py::class_<ov::runtime::CompiledModel, std::shared_ptr<ov::runtime::CompiledModel>> cls(m, "CompiledModel");

    cls.def(py::init([](ov::runtime::CompiledModel& other) {
                return other;
            }),
            py::arg("other"));

    cls.def("create_infer_request", [](ov::runtime::CompiledModel& self) {
        return InferRequestWrapper(self.create_infer_request(), self.inputs(), self.outputs());
    });

    cls.def(
        "infer_new_request",
        [](ov::runtime::CompiledModel& self, const py::dict& inputs) {
            auto request = self.create_infer_request();
            // Update inputs if there are any
            Common::set_request_tensors(request, inputs);
            request.infer();
            return Common::outputs_to_dict(self.outputs(), request);
        },
        py::arg("inputs"));

    cls.def("export_model", &ov::runtime::CompiledModel::export_model, py::arg("model_stream"));

    cls.def(
        "get_config",
        [](ov::runtime::CompiledModel& self, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_config(name)).as<py::object>();
        },
        py::arg("name"));

    cls.def(
        "get_metric",
        [](ov::runtime::CompiledModel& self, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_metric(name)).as<py::object>();
        },
        py::arg("name"));

    cls.def("get_runtime_model", &ov::runtime::CompiledModel::get_runtime_model);

    cls.def_property_readonly("inputs", &ov::runtime::CompiledModel::inputs);

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::CompiledModel::*)() const) & ov::runtime::CompiledModel::input);

    cls.def(
        "input",
        (ov::Output<const ov::Node>(ov::runtime::CompiledModel::*)(size_t) const) & ov::runtime::CompiledModel::input,
        py::arg("index"));

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::CompiledModel::*)(const std::string&) const) &
                ov::runtime::CompiledModel::input,
            py::arg("tensor_name"));

    cls.def_property_readonly("outputs", &ov::runtime::CompiledModel::outputs);

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::CompiledModel::*)() const) & ov::runtime::CompiledModel::output);

    cls.def(
        "output",
        (ov::Output<const ov::Node>(ov::runtime::CompiledModel::*)(size_t) const) & ov::runtime::CompiledModel::output,
        py::arg("index"));

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::CompiledModel::*)(const std::string&) const) &
                ov::runtime::CompiledModel::output,
            py::arg("tensor_name"));
}
