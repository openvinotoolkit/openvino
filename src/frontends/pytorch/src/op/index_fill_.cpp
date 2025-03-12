// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_index_fill_(const NodeContext& context) {
    // aten::index_fill_(self, dim, index, value) --> Tensor
    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.get_input(2);
    auto value = context.get_input(3);

    auto const_1_vec = v0::Constant::create(element::i32, Shape{1}, {1});

    auto tensor_rank = std::get<1>(get_shape_rank(context, input, false));
    auto tensor_rank_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(tensor_rank, dim));
    auto dim_vec = normalize_axis(context, dim, tensor_rank_correct_type);

    // scalar to vec
    auto value_vec = context.mark_node(std::make_shared<v1::Reshape>(value, const_1_vec, false));

    auto input_shape = std::get<0>(get_shape_rank(context, input, false));

    auto index_shape = std::get<0>(get_shape_rank(context, index, false));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto index_len = context.mark_node(std::make_shared<v8::Slice>(index_shape, const_0, const_1, const_1));

    // [A, B, ..., T, ..., K] --> [A, B, ..., len(index), ..., K]
    auto target_shape = std::make_shared<v12::ScatterElementsUpdate>(input_shape,
                                                                     dim_vec,
                                                                     index_len,
                                                                     v0::Constant::create(element::i32, Shape{}, {0}));

    // broadcast && index fill
    auto broadcasted_value = context.mark_node(std::make_shared<v1::Broadcast>(value_vec, target_shape, dim_vec));
    auto broadcasted_index = context.mark_node(std::make_shared<v1::Broadcast>(index, target_shape, dim_vec));
    auto result = context.mark_node(
        std::make_shared<v12::ScatterElementsUpdate>(input, broadcasted_index, broadcasted_value, dim));

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
