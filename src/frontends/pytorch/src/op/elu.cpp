// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_elu(const NodeContext& context) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto alpha = context.const_input<float>(1);
    auto elu = context.mark_node(std::make_shared<v0::Elu>(x, alpha));
    if (!context.input_is_none(2)) {
        auto scale = context.get_input(2);
        scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, elu));
        elu = context.mark_node(std::make_shared<v1::Multiply>(elu, scale));
    }
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(3) || context.const_input<int64_t>(3) == 1,
                                "Unexpected value of input_scale input for elu operation");
    return {elu};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov