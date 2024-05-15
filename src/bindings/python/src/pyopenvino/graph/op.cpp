// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pyopenvino/graph/op.hpp>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/op/op.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritence from TorchDecoder in Python
class PyOp : public ov::op::Op {
public:
    using ov::op::Op::Op;

    void validate_and_infer_types() override {
        PYBIND11_OVERRIDE(void, Op, validate_and_infer_types);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // PYBIND11_OVERRIDE_PURE(bool, Op, visit_attributes, visitor);
        //  Requires binding for visitor
        //  Now works only for operations without attributes
        return true;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Node>, Op, clone_with_new_inputs, new_args);
    }

    const type_info_t& get_type_info() const override {
        PYBIND11_OVERRIDE(const type_info_t&, ov::op::Op, get_type_info);
    }

    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const override {
        PYBIND11_OVERRIDE(bool, ov::op::Op, evaluate, output_values, input_values);
    }

    bool has_evaluate() const override {
        PYBIND11_OVERRIDE(bool, ov::op::Op, has_evaluate);
    }
};

void regclass_graph_Op(py::module m) {
    py::class_<ov::op::Op, std::shared_ptr<ov::op::Op>, PyOp, ov::Node>(m, "PyOp")
        .def(py::init<>());
}
