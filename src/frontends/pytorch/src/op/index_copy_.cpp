// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

std::tuple<Output<Node>, Output<Node>> get_transpose_perm(const NodeContext& context,
                                                          const Output<Node>& tensor_rank,
                                                          const Output<Node>& positive_dim,
                                                          const Output<Node>& positive_dim_plus1,
                                                          const std::shared_ptr<v0::Constant> const_0,
                                                          const std::shared_ptr<v0::Constant> const_1,
                                                          const std::shared_ptr<v0::Constant> const_neg_1) {
    // variable preparation
    auto const_0_vec = v0::Constant::create(element::i32, Shape{1}, {0});
    auto const_1_vec = v0::Constant::create(element::i32, Shape{1}, {1});
    auto positive_dim_vec = context.mark_node(std::make_shared<v1::Reshape>(positive_dim, const_1_vec, false));
    auto positive_dim_plus1_vec =
        context.mark_node(std::make_shared<v1::Reshape>(positive_dim_plus1, const_1_vec, false));

    auto tensor_rank_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(tensor_rank, positive_dim));
    OutputVector shapes_list;
    shapes_list.push_back(positive_dim_vec);
    auto internal_range =
        context.mark_node(std::make_shared<v4::Range>(const_0, positive_dim, const_1, element::i32));  // [0]
    auto internal_range_correct_type =
        context.mark_node(std::make_shared<v1::ConvertLike>(internal_range, positive_dim));
    shapes_list.push_back(internal_range_correct_type);
    Output<Node> perm = context.mark_node(std::make_shared<v0::Concat>(shapes_list, 0));  // [1, 0]
    auto diff =
        context.mark_node(std::make_shared<v1::Subtract>(tensor_rank_correct_type, positive_dim_plus1_vec));  // 1

    // compute the perm
    perm =
        context.mark_node(std::make_shared<v1::Pad>(perm, const_0_vec, diff, const_0, PadMode::CONSTANT));  // [1, 0, 0]
    auto negative_range = context.mark_node(
        std::make_shared<v4::Range>(const_0,
                                    context.mark_node(std::make_shared<v0::Negative>(positive_dim_plus1)),
                                    const_neg_1,
                                    element::i32));  // [0, -1]
    Output<Node> negative_range_correct_type =
        context.mark_node(std::make_shared<v1::ConvertLike>(negative_range, positive_dim));
    negative_range_correct_type = context.mark_node(std::make_shared<v1::Pad>(negative_range_correct_type,
                                                                              const_0_vec,
                                                                              diff,
                                                                              const_0,
                                                                              PadMode::CONSTANT));  // [0, -1, 0]
    auto full_range = context.mark_node(
        std::make_shared<v4::Range>(const_0, tensor_rank_correct_type, const_1, element::i32));  // [0, 1, 2]
    auto full_range_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(full_range, positive_dim));
    perm = context.mark_node(std::make_shared<v1::Add>(perm, negative_range_correct_type));
    perm = context.mark_node(std::make_shared<v1::Add>(perm, full_range_correct_type));  // [1, 0, 2]

    // compute the reverse perm
    OutputVector reverse_shapes_list;
    auto reverse_prefix =
        context.mark_node(std::make_shared<v4::Range>(const_1, positive_dim_plus1, const_1, element::i32));  // [1]
    auto reverse_prefix_correct_type =
        context.mark_node(std::make_shared<v1::ConvertLike>(reverse_prefix, positive_dim));
    reverse_shapes_list.push_back(reverse_prefix_correct_type);
    reverse_shapes_list.push_back(const_0_vec);
    Output<Node> reverse_perm = context.mark_node(std::make_shared<v0::Concat>(reverse_shapes_list, 0));  // [1, 0]
    reverse_perm = context.mark_node(
        std::make_shared<v1::Pad>(reverse_perm, const_0_vec, diff, const_0, PadMode::CONSTANT));  // [1, 0, 0]
    reverse_perm = context.mark_node(std::make_shared<v1::Add>(reverse_perm, negative_range_correct_type));
    reverse_perm = context.mark_node(std::make_shared<v1::Add>(reverse_perm, full_range_correct_type));  // [1, 0, 2]

    return std::make_tuple(perm, reverse_perm);
};

OutputVector translate_index_copy_(const NodeContext& context) {
    // aten::index_copy_(self, dim, index, tensor) → Tensor
    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.get_input(2);
    auto tensor = context.get_input(3);
    auto dim_scalar = context.const_input<int>(1);

    auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});
    auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});

    auto tensor_shape = context.mark_node(std::make_shared<v3::ShapeOf>(tensor, element::i32));  // [4, 2, 5]
    Output<Node> tensor_rank = context.mark_node(std::make_shared<v3::ShapeOf>(tensor_shape, element::i32));  // [3]
    tensor_rank = context.mark_node(std::make_shared<v8::Gather>(tensor_rank, const_0, const_0));             // 3
    auto tensor_rank_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(tensor_rank, dim));
    Output<Node> positive_dim = context.mark_node(std::make_shared<v1::Add>(dim, tensor_rank_correct_type));
    positive_dim = context.mark_node(std::make_shared<v1::Mod>(positive_dim, tensor_rank_correct_type));
    Output<Node> positive_dim_plus1 = context.mark_node(std::make_shared<v1::Add>(positive_dim, const_1));

    // get the correct index
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));  // [4, 3, 5]
    auto selected_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, positive_dim, const_0));  // 3
    auto selected_dim_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(selected_dim, index));
    Output<Node> correct_index = context.mark_node(std::make_shared<v1::Add>(index, selected_dim_correct_type));
    correct_index = context.mark_node(std::make_shared<v1::Mod>(correct_index, selected_dim_correct_type));
    auto unsqueezed_index =
        context.mark_node(std::make_shared<v0::Unsqueeze>(correct_index, const_neg_1));  // [[1], [0]]

    // begin the computation
    if (dim_scalar == 0) {
        // When dim == 0, the op is equavilent to ScatterNDUpdate
        auto result = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(input, unsqueezed_index, tensor));

        return {result};
    } else {
        // When dim > 0, we need to get correct tensors to use ScatterNDUpdate
        Output<Node> perm, reverse_perm;
        std::tie(perm, reverse_perm) =
            get_transpose_perm(context, tensor_rank, positive_dim, positive_dim_plus1, const_0, const_1, const_neg_1);
        auto transposed_tensor = context.mark_node(std::make_shared<v1::Transpose>(tensor, perm));  // [2, 4, 5]
        auto transposed_input = context.mark_node(std::make_shared<v1::Transpose>(input, perm));    // [3, 4, 5]
        auto result = context.mark_node(
            std::make_shared<v3::ScatterNDUpdate>(transposed_input, unsqueezed_index, transposed_tensor));  // [3, 4, 5]
        auto transposed_result = context.mark_node(std::make_shared<v1::Transpose>(result, reverse_perm));  // [4, 3, 5]

        return {transposed_result};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
