// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/model_pass.hpp"

#include <pybind11/pybind11.h>

#include <openvino/pass/pass.hpp>
#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/utils/utils.hpp"

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

void regclass_passes_ModelPass(py::module m) {
    py::class_<ov::pass::ModelPass, std::shared_ptr<ov::pass::ModelPass>, ov::pass::PassBase, PyModelPass> model_pass(
        m,
        "ModelPass");
    model_pass.doc() = "openvino.passes.ModelPass wraps ov::pass::ModelPass";
    model_pass.def(py::init<>());
    model_pass.def(
        "run_on_model",
        [](ov::pass::ModelPass& self, const py::object& ie_api_model) {
            const auto model = Common::utils::convert_to_model(ie_api_model);
            self.run_on_model(model);
        },
        py::arg("model"),
        R"(
                   run_on_model must be defined in inherited class. This method is used to work with Model directly.

                   :param model: openvino.Model to be transformed.
                   :type model: openvino.Model

                   :return: True in case if Model was changed and False otherwise.
                   :rtype: bool
    )");
    model_pass.def("__repr__", [](const ov::pass::ModelPass& self) {
        return Common::get_simple_repr(self);
    });
}
