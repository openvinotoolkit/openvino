// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_norm(NodeContext& context) {
    auto input_tensor = context.get_input(0);
    auto p = context.const_input<int64_t>(1);
    auto dim = context.get_input(2);
    auto keep_dim = context.const_input<bool>(3);

    std::shared_ptr<ov::op::util::ArithmeticReductionKeepDims> reduce;

    FRONT_END_OP_CONVERSION_CHECK(p == 1 || p == 2, "OpenVino supprots only p-norms with p of 1 or 2");

    if (p == 1) {
        reduce = std::make_shared<opset8::ReduceL1>(input_tensor, dim, keep_dim);
    } else if (p == 2) {
        reduce = std::make_shared<opset8::ReduceL2>(input_tensor, dim, keep_dim);
    }

    return {context.mark_node(reduce)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov