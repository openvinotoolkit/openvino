// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_prim_layout(const NodeContext& context) {
    // Check if one input is given
    num_inputs_check(context, 1, 1);

    // 0 = torch.strided layout
    auto layout_constant = std::make_shared<ov::opset10::Constant>(
        ov::element::i64,        // Data type: int64
        ov::Shape{},             // Scalar shape
        std::vector<int64_t>{0}  // Value: 0 = strided
    );

    return {context.mark_node(layout_constant)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
