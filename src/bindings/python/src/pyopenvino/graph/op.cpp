// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/op.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <pyopenvino/graph/op.hpp>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

namespace py = pybind11;

void PyOp::validate_and_infer_types() {
    PYBIND11_OVERRIDE(void, ov::op::Op, validate_and_infer_types);
}

bool PyOp::visit_attributes(ov::AttributeVisitor& value) {
    py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
    // Try to look up the overridden method on the Python side.
    py::function overrided_py_method = pybind11::get_override(this, "visit_attributes");
    if (overrided_py_method) {                                       // method is found
        return static_cast<py::bool_>(overrided_py_method(&value));  // Call the Python function.
    }
    return true;
}

std::shared_ptr<ov::Node> PyOp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
    // Try to look up the overridden method on the Python side.
    py::function overrided_py_method = pybind11::get_override(this, "clone_with_new_inputs");
    if (overrided_py_method) {                        // method is found
        auto result = overrided_py_method(new_args);  // Call the Python function.
        return result.cast<std::shared_ptr<ov::Node>>();
    }
    // Default implementation for clone_with_new_inputs
    auto py_handle_type = py_handle.get_type();
    auto new_py_object = py_handle_type(new_args);
    return new_py_object.cast<std::shared_ptr<ov::Node>>();
}

const ov::op::Op::type_info_t& PyOp::get_type_info() const {
    return *m_type_info;
}

bool PyOp::evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const {
    PYBIND11_OVERRIDE(bool, ov::op::Op, evaluate, output_values, input_values);
}

bool PyOp::has_evaluate() const {
    PYBIND11_OVERRIDE(bool, ov::op::Op, has_evaluate);
}

void regclass_graph_Op(py::module m) {
    py::class_<ov::op::Op, std::shared_ptr<ov::op::Op>, PyOp, ov::Node> op(m, "Op");

    op.def(py::init([](const py::object& py_obj) {
        return PyOp(py_obj);
    }));
}
