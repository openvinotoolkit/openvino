// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ie_input_info.hpp>

#include "common.hpp"
#include "openvino/runtime/executable_network.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/ie_infer_request.hpp"
#include "pyopenvino/core/ie_input_info.hpp"

namespace py = pybind11;

void regclass_ExecutableNetwork(py::module m) {
    py::class_<ov::runtime::ExecutableNetwork, std::shared_ptr<ov::runtime::ExecutableNetwork>> cls(
        m,
        "ExecutableNetwork");

    cls.def("create_infer_request", &ov::runtime::ExecutableNetwork::create_infer_request);

    cls.def("infer_new_request", [](ov::runtime::ExecutableNetwork& self, const py::dict& inputs) {
        // TODO: implment after https://github.com/openvinotoolkit/openvino/pull/7962
        // will be merged as a seperate ticket
    });

    cls.def("export_model", &ov::runtime::ExecutableNetwork::export_model);

    cls.def(
        "get_config",
        [](ov::runtime::ExecutableNetwork& self, const std::string& name) -> py::handle {
            return Common::parse_parameter(self.get_config(name));
        },
        py::arg("config_name"));

    cls.def(
        "get_metric",
        [](ov::runtime::ExecutableNetwork& self, const std::string& name) -> py::handle {
            return Common::parse_parameter(self.get_metric(name));
        },
        py::arg("metric_name"));

    cls.def("get_runtime_function", &ov::runtime::ExecutableNetwork::get_runtime_function);

    cls.def("inputs", &ov::runtime::ExecutableNetwork::inputs);

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)() const) &
                ov::runtime::ExecutableNetwork::input);

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(size_t) const) &
                ov::runtime::ExecutableNetwork::input);

    cls.def("input",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(const std::string&) const) &
                ov::runtime::ExecutableNetwork::input);

    cls.def("outputs", &ov::runtime::ExecutableNetwork::outputs);

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)() const) &
                ov::runtime::ExecutableNetwork::output);

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(size_t) const) &
                ov::runtime::ExecutableNetwork::output);

    cls.def("output",
            (ov::Output<const ov::Node>(ov::runtime::ExecutableNetwork::*)(const std::string&) const) &
                ov::runtime::ExecutableNetwork::output);

    cls.def("set_config", &ov::runtime::ExecutableNetwork::set_config);

    cls.def("get_context", &ov::runtime::ExecutableNetwork::get_context);
}
