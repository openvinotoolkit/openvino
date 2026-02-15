// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_masked_scatter(const NodeContext& context) {
    // aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
    // aten::masked_scatter.out(Tensor self, Tensor mask, Tensor source, *, Tensor(a!) out) -> Tensor(a!)
    // aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    auto x = context.get_input(0);
    auto mask = context.get_input(1);
    auto source = context.get_input(2);
    // mask should be broadcastable to x shape
    auto x_shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));
    auto expanded_mask =
        context.mark_node(std::make_shared<v3::Broadcast>(mask, x_shape, BroadcastType::BIDIRECTIONAL));
    auto index = context.mark_node(std::make_shared<v3::NonZero>(expanded_mask));
    auto input_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
    index = context.mark_node(std::make_shared<v1::Transpose>(index, input_order));
    // source can be arbitary shape, select only relevant data
    auto const_minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto flatten_source = context.mark_node(std::make_shared<v1::Reshape>(source, const_minus_one, false));
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto index_shape = context.mark_node(std::make_shared<v3::ShapeOf>(index, element::i32));
    auto index_dim_zero = context.mark_node(std::make_shared<v8::Gather>(index_shape, const_zero, const_zero));
    auto slice_steps = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto sliced_source = context.mark_node(
        std::make_shared<v8::Slice>(flatten_source, const_zero, index_dim_zero, slice_steps, const_zero));

    auto update = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(x, index, sliced_source));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, update);
    }
    return {update};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov