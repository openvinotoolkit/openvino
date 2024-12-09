// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/op_extension.hpp"

#include <pybind11/pybind11.h>

#include <pyopenvino/graph/op_extension.hpp>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/extension.hpp"
#include "pyopenvino/graph/discrete_type_info.hpp"
#include "pyopenvino/graph/node_output.hpp"
#include "pyopenvino/graph/op.hpp"

namespace py = pybind11;

void regclass_graph_OpExtension(py::module m) {
    py::class_<PyOpExtension, std::shared_ptr<PyOpExtension>, ov::Extension> op_extension(m, "OpExtension");
    op_extension.doc() = "openvino.OpExtension provides the base interface for OpenVINO extensions.";

    op_extension.def("__repr__", [](const PyOpExtension& self) {
        return Common::get_simple_repr(self);
    });

    op_extension.def(py::init([](py::object dtype) {
        py::object py_issubclass = py::module::import("builtins").attr("issubclass");
        if (py_issubclass(dtype, py::type::of<PyOp>())) {
            return PyOpExtension(dtype);
        }
        std::stringstream str;
        str << "Unsupported data type : '" << dtype << "' is passed as an argument.";
        OPENVINO_THROW(str.str());
    }));
}
