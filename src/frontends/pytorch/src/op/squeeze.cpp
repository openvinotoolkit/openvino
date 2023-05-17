// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_squeeze(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto input = context.get_input(0);
    if (context.input_is_none(1)) {
        return {context.mark_node(std::make_shared<v0::Squeeze>(input))};
    }
    // Cannot provide dimensions to ov v0::Squeeze directly due to mismatch in behavior between OV and PyTorch:
    // If provided dimension cannot be squeezed, OV raises exception, PyTorch returns dimension unmodified.
    auto dim_input = context.get_input(1);
    auto const_0 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    auto input_rank = context.mark_node(std::make_shared<v3::ShapeOf>(input_shape));
    auto input_rank_s = context.mark_node(std::make_shared<v0::Squeeze>(input_rank));
    // Create boolean mask containing True on 1, else 0.
    auto squeezable_dimensions_mask = context.mark_node(std::make_shared<v1::Equal>(input_shape, const_1));
    auto input_shape_indices =
        context.mark_node(std::make_shared<v4::Range>(const_0, input_rank_s, const_1, element::i64));
    // Translate negative dimension indices into positive only.
    auto dim_input_pos_only = context.mark_node(std::make_shared<v8::Gather>(input_shape_indices, dim_input, const_0));
    // Add additional dimension to axis indices, allowing to use broadcast in equal to create boolean mask,
    // where True indicates that input dimension was selected to be squeezed.
    auto dim_reshape_shape = context.mark_node(v0::Constant::create(element::i64, Shape{2}, {-1, 1}));
    auto reshaped_dim_input_pos_only =
        context.mark_node(std::make_shared<v1::Reshape>(dim_input_pos_only, dim_reshape_shape, false));
    auto selected_mask_to_squeeze =
        context.mark_node(std::make_shared<v1::Equal>(input_shape_indices, reshaped_dim_input_pos_only));
    selected_mask_to_squeeze =
        context.mark_node(std::make_shared<v1::ReduceLogicalOr>(selected_mask_to_squeeze, const_0));
    // Create mask indicating elements that are both selected to be squeezed, and are squeezable (have 1 dimension).
    auto dimension_mask_to_squeeze = context.mark_node(
        std::make_shared<v1::LogicalAnd>(selected_mask_to_squeeze, squeezable_dimensions_mask, "none"));
    // From input_shape, gather only those that shouldn't be squeezed, either because they weren't selected or were
    // unsqueezable.
    auto dimension_mask_to_preserve = context.mark_node(std::make_shared<v1::LogicalNot>(dimension_mask_to_squeeze));
    auto dimension_idxs_to_preserve = context.mark_node(std::make_shared<v3::NonZero>(dimension_mask_to_preserve));
    dimension_idxs_to_preserve = context.mark_node(std::make_shared<v0::Squeeze>(dimension_idxs_to_preserve));
    auto dimensions_to_preserve =
        context.mark_node(std::make_shared<v8::Gather>(input_shape, dimension_idxs_to_preserve, const_0));
    // Use reshape to remove dimensions that were selected and were squeezable.
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(input, dimensions_to_preserve, false));
    return {context.mark_node(reshape)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
