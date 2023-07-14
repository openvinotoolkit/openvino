// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_quantized_linear(const NodeContext& context) {
    // "quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i,
    // int Y_zero_point_i) -> Tensor Y" num_inputs_check(context, 4, 4);
    num_inputs_check(context, 4, 4);
    auto x = context.get_input(0);
    auto packed_params_input = context.get_input(1).get_node();
    auto packed_params = packed_params_input->inputs();
    auto weights = packed_params[0].get_source_output();
    auto bias = packed_params[1].get_source_output();

    auto linear = context.mark_node(std::make_shared<ov::op::v0::MatMul>(x, weights, false, true));
    linear = context.mark_node(std::make_shared<ov::op::v1::Add>(linear, bias));
    auto scale = context.get_input(2);
    auto zero_point = context.get_input(3);
    return {context.mark_output(quantize(context, linear, scale, zero_point, context.get_input(0)))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
