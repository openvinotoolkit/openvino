// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/internal/gqa_extension.hpp"

#include "openvino/core/op_extension.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_GroupQueryAttention(py::module m) {
    using ov::op::internal::GroupQueryAttention;

    py::class_<ov::OpExtension<GroupQueryAttention>,
               std::shared_ptr<ov::OpExtension<GroupQueryAttention>>,
               ov::Extension>(m, "_GroupQueryAttentionExtension")
        .def(py::init<>())
        .doc() = "Extension that registers GroupQueryAttention for IR deserialization. "
                 "Pass an instance to core.add_extension() before reading an IR that contains this op.";
}
