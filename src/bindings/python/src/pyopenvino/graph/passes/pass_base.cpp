// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/pass.hpp>
#include <pybind11/pybind11.h>

#include <memory>

#include "pyopenvino/graph/passes/pass_base.hpp"

namespace py = pybind11;


void regclass_PassBase(py::module m) {
    py::class_<ov::pass::PassBase, std::shared_ptr<ov::pass::PassBase>> pass_base(m, "PassBase");
    pass_base.doc() = "openvino.runtime.passes.PassBase wraps ov::pass::PassBase";
    pass_base.def("set_name", &ov::pass::PassBase::set_name);
    pass_base.def("get_name", &ov::pass::PassBase::get_name);
}
