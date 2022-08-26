// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs mul(const NodeContext& node) {
    auto x = node.get_input("X");
    auto y = node.get_input("Y");
    PADDLE_OP_CHECK(node, x.get_partial_shape().rank().is_static(), "matmul: X rank must be static!");
    int64_t x_rank = x.get_partial_shape().rank().get_length();
    PADDLE_OP_CHECK(node,
                    y.get_partial_shape().rank().is_static() && y.get_partial_shape().rank().get_length() == 2,
                    "matmul: Y rank must be static, and 2!");
    if (x_rank > 2) {
        auto shape = std::make_shared<default_opset::ShapeOf>(x);
        int64_t x_num_col_dims = node.get_attribute<int32_t>("x_num_col_dims");
        auto axis = default_opset::Constant::create(ngraph::element::i64, {}, {0});
        auto split_lengths =
            default_opset::Constant::create(ngraph::element::i64, {2}, {x_num_col_dims, x_rank - x_num_col_dims});
        auto split = std::make_shared<default_opset::VariadicSplit>(shape, axis, split_lengths);
        auto f_dim_red_axis = default_opset::Constant::create(ngraph::element::i64, {}, {0});
        auto first_dim_reduce = std::make_shared<default_opset::ReduceProd>(split->output(0), f_dim_red_axis);
        auto f_dim_shape = default_opset::Constant::create(ngraph::element::i64, {1}, {1});
        auto first_dim = std::make_shared<default_opset::Reshape>(first_dim_reduce, f_dim_shape, false);
        auto s_dim_red_axis = default_opset::Constant::create(ngraph::element::i64, {}, {0});
        auto second_dim_reduce = std::make_shared<default_opset::ReduceProd>(split->output(1), s_dim_red_axis);
        auto s_dim_shape = default_opset::Constant::create(ngraph::element::i64, {1}, {1});
        auto second_dim = std::make_shared<default_opset::Reshape>(second_dim_reduce, s_dim_shape, false);
        auto out_shape = std::make_shared<default_opset::Concat>(ngraph::NodeVector{first_dim, second_dim}, 0);
        auto x_reshaped = std::make_shared<default_opset::Reshape>(x, out_shape, false);
        return node.default_single_output_mapping({std::make_shared<default_opset::MatMul>(x_reshaped, y)}, {"Out"});
    }
    return node.default_single_output_mapping({std::make_shared<default_opset::MatMul>(x, y)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
