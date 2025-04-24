// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector create_argmax_argmin_op(const NodeContext& context, TopKMode mode) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    bool keep_dims = false;
    auto k = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    Output<Node> indices;
    if (!context.input_is_none(2)) {
        keep_dims = context.const_input<bool>(2);
    }
    if (!context.input_is_none(1)) {
        auto axis = context.const_input<int64_t>(1);
        auto topk = context.mark_node(
            std::make_shared<v11::TopK>(input, k, axis, mode, TopKSortType::SORT_VALUES, element::i64, true));
        indices = topk->output(1);
        if (!keep_dims) {
            auto axis_to_remove = context.mark_node(v0::Constant::create(element::i32, Shape{}, {axis}));
            indices = context.mark_node(std::make_shared<v0::Squeeze>(indices, axis_to_remove));
        }
    } else {
        int64_t axis = 0;
        auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
        auto flatten_input = context.mark_node(std::make_shared<v1::Reshape>(input, minus_one, false));
        auto topk = context.mark_node(
            std::make_shared<v11::TopK>(flatten_input, k, axis, mode, TopKSortType::SORT_VALUES, element::i64, true));
        indices = topk->output(1);
        if (keep_dims) {
            auto input_rank = std::get<1>(get_shape_rank(context, input));
            auto new_shape = context.mark_node(std::make_shared<v3::Broadcast>(k, input_rank));
            indices =
                context.mark_node(std::make_shared<v3::Broadcast>(indices, new_shape, BroadcastType::BIDIRECTIONAL));
        } else {
            auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
            indices = context.mark_node(std::make_shared<v0::Squeeze>(indices, zero));
        }
    }
    return {std::move(indices)};
}

};  // namespace

OutputVector translate_argmax(const NodeContext& context) {
    // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
    return create_argmax_argmin_op(context, TopKMode::MAX);
};

OutputVector translate_argmin(const NodeContext& context) {
    // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
    return create_argmax_argmin_op(context, TopKMode::MIN);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
