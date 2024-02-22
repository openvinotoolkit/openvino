// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_elu(const NodeContext& context) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto alpha = context.const_input<float>(1);
    // TODO: Figure out what scale and input_scale do
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<int64_t>(2) == 1,
                                "Unexpected value of scale input for elu operation");
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(3) || context.const_input<int64_t>(3) == 1,
                                "Unexpected value of input_scale input for elu operation");
    return {context.mark_node(std::make_shared<ov::op::v0::Elu>(x, alpha))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov