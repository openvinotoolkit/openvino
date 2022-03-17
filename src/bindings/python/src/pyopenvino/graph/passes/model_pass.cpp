// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/model_pass.hpp"

#include <pybind11/pybind11.h>

#include <openvino/pass/pass.hpp>
#include <string>

namespace py = pybind11;

class PyModelPass : public ov::pass::ModelPass {
public:
    /* Inherit the constructors */
    using ov::pass::ModelPass::ModelPass;

    /* Trampoline (need one for each virtual function) */
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        PYBIND11_OVERRIDE_PURE(bool,                /* Return type */
                               ov::pass::ModelPass, /* Parent class */
                               run_on_model,        /* Name of function in C++ (must match Python name) */
                               model                /* Argument(s) */
        );
    }
};

void regclass_ModelPass(py::module m) {
    py::class_<ov::pass::ModelPass, std::shared_ptr<ov::pass::ModelPass>, ov::pass::PassBase, PyModelPass> model_pass(
        m,
        "ModelPass");
    model_pass.doc() = "openvino.runtime.passes.ModelPass wraps ov::pass::ModelPass";
    model_pass.def(py::init<>());
    model_pass.def("run_on_model", &ov::pass::ModelPass::run_on_model);
}
