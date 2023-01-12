// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_batch_norm(NodeContext& context) {
    // Schema: aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var,
    // bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
    auto input = context.get_input(0);
    auto weight = context.get_input(1);
    auto bias = context.get_input(2);
    auto running_mean = context.get_input(3);
    auto running_var = context.get_input(4);
    auto training = context.const_input<bool>(5);
    FRONT_END_OP_CONVERSION_CHECK(!training, "Translation for aten::batch_norm do not support training mode.");
    // Index with index 6 is momentum, it is used only in training mode
    auto epsilon = context.const_input<float>(7);
    return {context.mark_node(
        std::make_shared<opset8::BatchNormInference>(input, weight, bias, running_mean, running_var, epsilon))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov