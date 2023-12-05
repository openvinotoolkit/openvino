// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
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
    auto eps = static_cast<float>(context.const_input<double>(eps_idx));
    Output<Node> input_shape;
    Output<Node> input_rank;
    std::tie(input_shape, input_rank) = get_shape_rank(context, data, true, element::i32);
    auto scalar_one = context.mark_node(v0::Constant::create(element::i32, {}, {1}));
    auto shape = context.mark_node(
        std::make_shared<v0::Constant>(element::i32, Shape({3}), std::vector<int64_t>{0, num_groups, -1}));
    auto reshaped_input = context.mark_node(std::make_shared<v1::Reshape>(data, shape, true));
    auto reduction_axes = context.mark_node(v0::Constant::create(element::i32, Shape({1}), std::vector<int64_t>(1, 2)));
    auto reshaped_norm = context.mark_node(
        std::make_shared<v6::MVN>(reshaped_input, reduction_axes, true, eps, MVNEpsMode::INSIDE_SQRT));
    auto norm = context.mark_node(std::make_shared<v1::Reshape>(reshaped_norm, input_shape, true));
    auto skip_last = context.mark_node(std::make_shared<v1::Subtract>(input_rank, scalar_one));
    auto axes = context.mark_node(std::make_shared<v4::Range>(scalar_one, skip_last, scalar_one, element::i32));
    if (!context.input_is_none(weights_idx)) {
        auto weights = context.get_input(static_cast<int>(weights_idx));
        weights = context.mark_node(std::make_shared<v0::Unsqueeze>(weights, axes));
        norm = context.mark_node(std::make_shared<v1::Multiply>(norm, weights));
    }
    if (!context.input_is_none(bias_idx)) {
        auto bias = context.get_input(static_cast<int>(bias_idx));
        bias = context.mark_node(std::make_shared<v0::Unsqueeze>(bias, axes));
        norm = context.mark_node(std::make_shared<v1::Add>(norm, bias));
    }
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
