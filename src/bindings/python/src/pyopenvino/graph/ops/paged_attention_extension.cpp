// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/paged_attention_extension.hpp"

#include "openvino/op/op.hpp"
#include "openvino/op/paged_attention.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_PagedAttentionExtension(py::module m) {
    using ov::op::PagedAttentionExtension;
    py::class_<PagedAttentionExtension, std::shared_ptr<PagedAttentionExtension>, ov::Node> cls(
        m,
        "_PagedAttentionExtension");
    cls.doc() = "Experimental extention for PagedAttention operation. Use with care: no backward compatibility is "
                "guaranteed in future releases.";
    cls.def(py::init<const ov::OutputVector&>());
}
