// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> generate_zeros_with_convertlike(const NodeContext& context,
                                             const Output<Node> sizes,
                                             const Output<Node> tensor_of_type) {
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(const_0, sizes));
    return context.mark_node(std::make_shared<v1::ConvertLike>(zeros, tensor_of_type));
}
}  // namespace

OutputVector translate_index_put_(NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_max_int =
        context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));

    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto indices = context.get_input(1);
    auto values = context.get_input(2);
    auto accumulate = context.const_input<bool>(3);

    auto indices_partial_shape = indices.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(indices_partial_shape.rank().is_static(),
                                  "We support only indices with static rank.");
    auto indices_first_dim = indices_partial_shape[0];
    FRONT_END_OP_CONVERSION_CHECK(indices_first_dim.is_static(),
                                  "We support only lists of tensors with static number of elements.");
    int64_t indices_list_len = indices_first_dim.get_length();
    if (indices_list_len == 0) {
        return {values};
    }

    auto const_indices_list_len = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {indices_list_len}));
    auto split_indices = context.mark_node(std::make_shared<v1::Split>(indices, const_0, indices_list_len));

    std::shared_ptr<Node> broadcast_index_shape;
    Output<Node> index;
    if (indices_list_len > 1) {
        index = split_indices->output(0);
        for (int i = 1; i < indices_list_len; i++) {
            index = context.mark_node(std::make_shared<v1::Add>(index, split_indices->output(i)));
        }
        broadcast_index_shape = context.mark_node(std::make_shared<v3::ShapeOf>(index, element::i32));
        OutputVector indices_list;
        for (int i = 0; i < indices_list_len; i++) {
            auto broadcast =
                context.mark_node(std::make_shared<v3::Broadcast>(split_indices->output(i), broadcast_index_shape));
            auto unsqueeze = context.mark_node(std::make_shared<v0::Unsqueeze>(broadcast, const_neg_1));

            // change negative indices to positive indices
            auto const_i = context.mark_node(v0::Constant::create(element::i32, Shape{}, {i}));
            auto dim_i = context.mark_node(std::make_shared<v8::Gather>(input_shape, const_i, const_0));
            auto dim_i_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(dim_i, index));
            unsqueeze = context.mark_node(std::make_shared<v1::Add>(unsqueeze, dim_i_correct_type));
            unsqueeze = context.mark_node(std::make_shared<v1::Mod>(unsqueeze, dim_i_correct_type));

            indices_list.push_back(unsqueeze);
        }
        index = context.mark_node(std::make_shared<v0::Concat>(indices_list, -1));
    } else {
        index = split_indices->output(0);

        // change negative indices to positive indices
        auto dim_0 = context.mark_node(std::make_shared<v8::Gather>(input_shape, const_0, const_0));
        auto dim_0_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(dim_0, index));
        index = context.mark_node(std::make_shared<v1::Add>(index, dim_0_correct_type));
        index = context.mark_node(std::make_shared<v1::Mod>(index, dim_0_correct_type));

        broadcast_index_shape = context.mark_node(std::make_shared<v3::ShapeOf>(index, element::i32));
        index = context.mark_node(std::make_shared<v0::Unsqueeze>(index, const_neg_1));
    }

    auto sub_data_shape =
        context.mark_node(std::make_shared<v8::Slice>(input_shape, const_indices_list_len, const_max_int, const_1));
    auto values_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{broadcast_index_shape, sub_data_shape}, 0));
    values = context.mark_node(std::make_shared<v3::Broadcast>(values, values_shape));
    values = context.mark_node(std::make_shared<v1::ConvertLike>(values, input));

    Output<Node> result;
    if (accumulate) {
        auto zeros = generate_zeros_with_convertlike(context, input_shape, input);
        result = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(zeros, index, values));
        result = context.mark_node(std::make_shared<v1::Add>(input, result));
    } else {
        result = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(input, index, values));
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov