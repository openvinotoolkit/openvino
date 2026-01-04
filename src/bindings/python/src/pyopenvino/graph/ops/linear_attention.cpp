// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/linear_attention.hpp"

#include "openvino/op/op.hpp"
#include "openvino/op/linear_attn.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_LinearAttention(py::module m) {
    using ov::op::LinearAttention;
    py::class_<LinearAttention, std::shared_ptr<LinearAttention>, ov::Node> cls(
        m,
        "_LinearAttention");
    cls.doc() = "Experimental extention for LinearAttention operation. Use with care: no backward compatibility is "
                "guaranteed in future releases.";
    cls.def(py::init<const ov::OutputVector&>());
}
