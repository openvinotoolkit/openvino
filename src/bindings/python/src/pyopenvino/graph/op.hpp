// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/op/op.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritance from TorchDecoder in Python
class PyOp : public ov::op::Op {
public:
    using ov::op::Op::Op;

    // Keeps a reference to the Python object to manage its lifetime
    PyOp(const py::object& py_obj) : py_handle(py_obj) {}

    void validate_and_infer_types() override;

    bool visit_attributes(ov::AttributeVisitor& value) override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const type_info_t& get_type_info() const override;

    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const override;

    bool has_evaluate() const override;

private:
    py::object py_handle;  // Holds the Python object to manage its lifetime
};

void regclass_graph_Op(py::module m);
