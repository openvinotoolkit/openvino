// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"
#include "pyopenvino/graph/passes/manager.hpp"

namespace py = pybind11;

namespace {
class ManagerWrapper : public ov::pass::Manager {
public:
    ManagerWrapper() {}
    ~ManagerWrapper() {}
    void register_pass(std::string pass_name) {
        if (pass_name == "ConstantFolding")
            push_pass<ov::pass::ConstantFolding>();

        if (m_per_pass_validation) {
            push_pass<ov::pass::Validate>();
        }
        return;
    }
};
}  // namespace

void regclass_graph_passes_Manager(py::module m) {
    py::class_<ManagerWrapper> manager(m, "Manager");
    manager.doc() = "ngraph.impl.passes.Manager wraps ov::pass::Manager using ManagerWrapper";

    manager.def(py::init<>());

    manager.def("set_per_pass_validation", &ManagerWrapper::set_per_pass_validation);
    manager.def("run_passes", &ManagerWrapper::run_passes);
    manager.def("register_pass", (void (ManagerWrapper::*)(std::string)) & ManagerWrapper::register_pass);
}
