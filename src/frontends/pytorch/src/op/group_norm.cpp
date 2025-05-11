// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_group_norm_common(const NodeContext& context,
                                         const size_t max_inputs,
                                         const size_t group_idx,
                                         const size_t eps_idx,
                                         const size_t weights_idx,
                                         const size_t bias_idx) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float
    // eps=1.0000000000000001e-05, bool cudnn_enabled=True) -> Tensor
    num_inputs_check(context, 2, max_inputs);
    auto data = context.get_input(0);
    auto num_groups = context.const_input<int64_t>(group_idx);
    // input 2 - weights and input 3 - bias are optional without default value, we handle them later
    auto eps = context.const_input<double>(eps_idx);

    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(data, element::i32));
    auto channels = context.mark_node(std::make_shared<v8::Gather>(shape, one, zero));
    channels = context.mark_node(std::make_shared<v0::Unsqueeze>(channels, zero));

    Output<Node> scale;
    if (!context.input_is_none(weights_idx)) {
        scale = context.get_input(static_cast<int>(weights_idx));
    } else {
        scale = context.mark_node(std::make_shared<v3::Broadcast>(one, channels));
        scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, data));
    }
    Output<Node> bias;
    if (!context.input_is_none(bias_idx)) {
        bias = context.get_input(static_cast<int>(bias_idx));
    } else {
        bias = context.mark_node(std::make_shared<v3::Broadcast>(zero, channels));
        bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, data));
    }
    auto norm = context.mark_node(std::make_shared<v12::GroupNormalization>(data, scale, bias, num_groups, eps));
    // Input with index 5 is flag "cudnn_enabled" we can ignore it
    return {norm};
};

OutputVector translate_group_norm(const NodeContext& context) {
    return translate_group_norm_common(context, 6, 1, 4, 2, 3);
}

OutputVector translate_group_norm_fx(const NodeContext& context) {
    auto output = translate_group_norm_common(context, 8, 6, 7, 1, 2);
    return {context.mark_node(make_list_construct(output))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
