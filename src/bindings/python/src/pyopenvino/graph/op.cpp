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

/// Trampoline class to support inheritance from TorchDecoder in Python
class PyOp : public ov::op::Op {
public:
    using ov::op::Op::Op;

    // Keeps a reference to the Python object to manage its lifetime
    PyOp(const py::object& py_obj) : py_handle(py_obj) {}

    void validate_and_infer_types() override {
        PYBIND11_OVERRIDE(void, ov::op::Op, validate_and_infer_types);
    }

    bool visit_attributes(ov::AttributeVisitor& value) override {
        pybind11::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
        // Try to look up the overridden method on the Python side.
        pybind11::function overrided_py_method = pybind11::get_override(this, "visit_attributes");
        if (overrided_py_method) {                                       // method is found
            return static_cast<py::bool_>(overrided_py_method(&value));  // Call the Python function.
        }
        return false;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Node>, ov::op::Op, clone_with_new_inputs, new_args);
    }

    const type_info_t& get_type_info() const override {
        PYBIND11_OVERRIDE(const ov::Node::type_info_t&, ov::op::Op, get_type_info);
    }

    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const override {
        PYBIND11_OVERRIDE(bool, ov::op::Op, evaluate, output_values, input_values);
    }

    bool has_evaluate() const override {
        PYBIND11_OVERRIDE(bool, ov::op::Op, has_evaluate);
    }

private:
    py::object py_handle;  // Holds the Python object to manage its lifetime
};

void regclass_graph_Op(py::module m) {
    py::class_<ov::op::Op, std::shared_ptr<ov::op::Op>, PyOp, ov::Node> op(m, "Op");

    op.def(py::init([](const py::object& py_obj) {
        return PyOp(py_obj);
    }));
}
