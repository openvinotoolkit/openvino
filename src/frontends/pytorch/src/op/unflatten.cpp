// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_unflatten(const NodeContext& context) {
    // aten::unflatten.int(Tensor(a) self, int dim, int[] sizes) -> Tensor(a)
    num_inputs_check(context, 3, 3);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto sizes = context.get_input(2);
    if (context.get_input_type(2).is<type::List>()) {
        sizes = concat_list_construct(sizes);
    }
    Output<Node> input_shape;
    Output<Node> rank;
    std::tie(input_shape, rank) = get_shape_rank(context, input);
    auto zero_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto one_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
    dim = normalize_axis(context, dim, rank);
    sizes = context.mark_node(std::make_shared<v0::Convert>(sizes, element::i32));
    auto max_int = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int>::max()}));
    auto dim_plus_one = context.mark_node(std::make_shared<v1::Add>(dim, one_1d));
    auto head_part_rank = context.mark_node(std::make_shared<v8::Slice>(input_shape, zero_1d, dim, one_1d));
    auto tail_part_rank = context.mark_node(std::make_shared<v8::Slice>(input_shape, dim_plus_one, max_int, one_1d));
    auto new_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{head_part_rank, sizes, tail_part_rank}, 0));
    return {context.mark_node(std::make_shared<v1::Reshape>(input, new_shape, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov