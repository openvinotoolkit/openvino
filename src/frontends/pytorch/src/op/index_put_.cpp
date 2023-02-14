// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

namespace {
ov::Output<Node> generate_zeros_with_convertlike(NodeContext& context,
                                                 Output<Node> sizes,
                                                 Output<Node> tensor_of_type) {
    auto const_0 = context.mark_node(Constant::create(element::i32, Shape{}, {0}));
    auto zeros = context.mark_node(std::make_shared<Broadcast>(const_0, sizes));
    return context.mark_node(std::make_shared<ConvertLike>(zeros, tensor_of_type));
}
}  // namespace

OutputVector translate_index_put_(NodeContext& context) {
    auto const_0 = context.mark_node(Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(Constant::create(element::i32, Shape{1}, {1}));
    auto const_max_int =
        context.mark_node(Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()}));
    auto const_neg_1 = context.mark_node(Constant::create(element::i32, Shape{}, {-1}));

    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<ShapeOf>(input, element::i32));
    auto indices = context.get_input(1);
    auto values = context.get_input(2);
    auto accumulate = context.const_input<bool>(3);

    auto indices_partial_shape = indices.get_partial_shape();
    auto indices_first_dim = indices_partial_shape[0];
    FRONT_END_OP_CONVERSION_CHECK(indices_first_dim.is_static(),
                                  "We support only lists of tensors with constant number of elements.");
    int64_t indices_list_len = indices_first_dim.get_length();
    if (indices_list_len == 0) {
        return {values};
    }

    auto const_indices_list_len = context.mark_node(Constant::create(element::i32, Shape{1}, {indices_list_len}));
    auto split_indices = context.mark_node(std::make_shared<Split>(indices, const_0, indices_list_len));

    std::shared_ptr<ov::Node> broadcast_index_shape;
    ov::Output<ov::Node> index;
    if (indices_list_len > 1) {
        index = split_indices->output(0);
        for (int i = 1; i < indices_list_len; i++) {
            index = context.mark_node(std::make_shared<Add>(index, split_indices->output(i)));
        }
        broadcast_index_shape = std::make_shared<ShapeOf>(index, element::i32);
        OutputVector indices_list;
        for (int i = 0; i < indices_list_len; i++) {
            auto broadcast =
                context.mark_node(std::make_shared<Broadcast>(split_indices->output(i), broadcast_index_shape));
            auto unsqueeze = context.mark_node(std::make_shared<Unsqueeze>(broadcast, const_neg_1));
            indices_list.push_back(unsqueeze);
        }
        index = context.mark_node(std::make_shared<Concat>(indices_list, -1));
    } else {
        index = split_indices->output(0);
        broadcast_index_shape = context.mark_node(std::make_shared<ShapeOf>(index, element::i32));
        index = context.mark_node(std::make_shared<Unsqueeze>(index, const_neg_1));
    }

    auto sub_data_shape =
        context.mark_node(std::make_shared<Slice>(input_shape, const_indices_list_len, const_max_int, const_1));
    auto values_shape =
        context.mark_node(std::make_shared<Concat>(OutputVector{broadcast_index_shape, sub_data_shape}, 0));
    values = context.mark_node(std::make_shared<Broadcast>(values, values_shape));
    values = context.mark_node(std::make_shared<ConvertLike>(values, input));

    std::shared_ptr<ov::Node> result;
    if (accumulate) {
        auto zeros = generate_zeros_with_convertlike(context, input_shape, input);
        result = context.mark_node(std::make_shared<ScatterNDUpdate>(zeros, index, values));
        result = context.mark_node(std::make_shared<Add>(input, result));
    } else {
        result = context.mark_node(std::make_shared<ScatterNDUpdate>(input, index, values));
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov