// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "node_context.hpp"
#include "openvino/frontend/node_context.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_NodeContext(py::module m) {
    py::class_<ov::frontend::NodeContext<ov::OutputVector>,
            std::shared_ptr<ov::frontend::NodeContext<ov::OutputVector>>>
            ext(m, "NodeContext", py::dynamic_attr());

    // todo: use another name?
    py::class_<ov::frontend::NodeContext<std::map<std::string, ov::OutputVector>>,
            std::shared_ptr<ov::frontend::NodeContext<std::map<std::string, ov::OutputVector>>>>
            ext_2(m, "NodeContext", py::dynamic_attr());
}