// Copyright (C) 2018-2023 Intel Corporation
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
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_index_fill_(const NodeContext& context) {

    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.get_input(2);
    auto value = context.get_input(3);

    auto const_1_vec = v0::Constant::create(element::i32, Shape{1}, {value});

    Output<Node> tensor_rank = std::get<1>(get_shape_rank(context, input, true));
    auto tensor_rank_correct_type = context.mark_node(std::make_shared<v1::ConvertLike>(tensor_rank, dim));
    auto positive_dim = normalize_axis(context, dim, tensor_rank_correct_type);


    auto tensor_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto dim_vec = context.mark_node(std::make_shared<v1::Reshape>(positive_dim, const_1_vec, false));
    auto broadcasted_index = context.mark_node(std::make_shared<v1::Broadcast>(index, tensor_shape, dim_vec));


    
    
    auto result =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(input, broadcasted_index, index, dim));
    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
