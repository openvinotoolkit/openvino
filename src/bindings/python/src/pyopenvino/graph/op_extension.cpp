// Copyright (C) 2018-2025 Intel Corporation
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

const ov::DiscreteTypeInfo& PyOpExtension::get_type_info() const {
    return *m_type_info;
}

ov::OutputVector PyOpExtension::create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const {
    py::gil_scoped_acquire acquire;

    const auto node = py_handle_dtype();

    node.attr("set_arguments")(py::cast(inputs));
    if (node.attr("visit_attributes")(&visitor)) {
        node.attr("constructor_validate_and_infer_types")();
    }

    return py::cast<ov::OutputVector>(node.attr("outputs")());
}

std::vector<ov::Extension::Ptr> PyOpExtension::get_attached_extensions() const {
    return {};
}

void regclass_graph_OpExtension(py::module m) {
    py::class_<PyOpExtension, std::shared_ptr<PyOpExtension>, ov::Extension> op_extension(m, "OpExtension");
    op_extension.doc() = "openvino.OpExtension provides the base interface for OpenVINO extensions.";

    op_extension.def("__repr__", [](const PyOpExtension& self) {
        return Common::get_simple_repr(self);
    });

    op_extension.def(py::init([](py::object dtype) {
        return PyOpExtension(dtype);
    }));
}
