// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_take_along_dim(const NodeContext& context) {
    // aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
    // aten::take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    auto x = context.get_input(0);
    auto index = context.get_input(1);
    index = context.mark_node(std::make_shared<ov::op::v0::Convert>(index, element::i32));
    int64_t axis = 0;

    if (context.input_is_none(2)) {
        // if dimension is not provided, flattenize input first
        auto minus_1 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {-1}));
        x = context.mark_node(std::make_shared<ov::op::v1::Reshape>(x, minus_1, false));
    } else {
        axis = context.const_input<int64_t>(2);
        // OpenVINO GatherElements requires to have equal dims between index and input except specified axis
        // while PyTorch requires to have them broadcastable
        auto axis_node = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {axis}));
        auto const_1 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {1}));
        auto const_0 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
        auto x_shape = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(x, element::i32));
        auto broadcast_shape =
            context.mark_node(std::make_shared<ov::op::v3::ScatterUpdate>(x_shape, axis_node, const_1, const_0));
        index = context.mark_node(
            std::make_shared<ov::op::v3::Broadcast>(index, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL));
    }
    auto gather_elements = context.mark_node(std::make_shared<ov::op::v6::GatherElements>(x, index, axis));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, gather_elements);
    }
    return {gather_elements};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov