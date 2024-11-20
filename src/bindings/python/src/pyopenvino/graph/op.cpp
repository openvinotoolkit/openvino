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

bool  PyOp::visit_attributes(ov::AttributeVisitor& value) {
    py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
    // Try to look up the overridden method on the Python side.
    py::function overrided_py_method = pybind11::get_override(this, "visit_attributes");
    if (overrided_py_method) {                                       // method is found
        return static_cast<py::bool_>(overrided_py_method(&value));  // Call the Python function.
    }
    return false;
}

std::shared_ptr<ov::Node>  PyOp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Node>, ov::op::Op, clone_with_new_inputs, new_args);
}

const ov::op::Op::type_info_t& PyOp::get_type_info() const {
    PYBIND11_OVERRIDE(const ov::Node::type_info_t&, ov::op::Op, get_type_info);
}

bool  PyOp::evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const {
    PYBIND11_OVERRIDE(bool, ov::op::Op, evaluate, output_values, input_values);
}

bool  PyOp::has_evaluate() const {
    PYBIND11_OVERRIDE(bool, ov::op::Op, has_evaluate);
}

void regclass_graph_Op(py::module m) {
    py::class_<ov::op::Op, std::shared_ptr<ov::op::Op>, PyOp, ov::Node> op(m, "Op");

    op.def(py::init([](const py::object& py_obj) {
        return PyOp(py_obj);
    }));
}
