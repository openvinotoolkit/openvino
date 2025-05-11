// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <pybind11/pybind11.h>

#include <utility>

#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/serialize.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/passes/manager.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using Version = ov::pass::Serialize::Version;
using FilePaths = std::pair<const std::string, const std::string>;

void regclass_passes_Manager(py::module m) {
    py::class_<ov::pass::Manager> manager(m, "Manager");
    manager.doc() = "openvino.passes.Manager executes sequence of transformation on a given Model";

    manager.def(py::init<>());
    manager.def("set_per_pass_validation",
                &ov::pass::Manager::set_per_pass_validation,
                py::arg("new_state"),
                R"(
                Enables or disables Model validation after each pass execution.

                :param new_state: flag which enables or disables model validation.
                :type new_state: bool
    )");

    manager.def(
        "run_passes",
        [](ov::pass::Manager& self, const py::object& ie_api_model) {
            const auto model = Common::utils::convert_to_model(ie_api_model);
            self.run_passes(model);
        },
        py::arg("model"),
        R"(
                Executes sequence of transformations on given Model.

                :param model: openvino.Model to be transformed.
                :type model: openvino.Model
    )");

    manager.def("register_pass",
                &ov::pass::Manager::register_pass_instance,
                py::arg("transformation"),
                R"(
                Register pass instance for execution. Execution order matches the registration order.

                :param transformation: transformation instance.
                :type transformation: openvino.passes.PassBase
    )");

    manager.def("__repr__", [](const ov::pass::Manager& self) {
        return Common::get_simple_repr(self);
    });
}
