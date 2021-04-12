//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "utility.hpp"
#include "mul.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

OutputVector mul (const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto y = node.get_ng_input("Y");
    MY_ASSERT(x.get_partial_shape().rank().is_static(), "matmul: X rank must be static!");
    int64_t x_rank = x.get_partial_shape().rank().get_length();
    MY_ASSERT(y.get_partial_shape().rank().is_static() &&
              y.get_partial_shape().rank().get_length() == 2, "matmul: Y rank must be static, and 2!");
    if (x_rank > 2) {
        auto shape = std::make_shared<ngraph::opset6::ShapeOf>(x);
        int64_t x_num_col_dims = node.get_attribute<int32_t>("x_num_col_dims");
        auto axis = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0});
        auto split_lengths = ngraph::opset6::Constant::create(ngraph::element::i64, {2},
                                                              {x_num_col_dims, x_rank - x_num_col_dims});
        auto split = std::make_shared<ngraph::opset6::VariadicSplit>(shape, axis, split_lengths);
        auto f_dim_red_axis = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0});
        auto first_dim_reduce = std::make_shared<ngraph::opset6::ReduceProd>(split->output(0),
                                                                             f_dim_red_axis);
        auto f_dim_shape = ngraph::opset6::Constant::create(ngraph::element::i64, {1}, {1});
        auto first_dim = std::make_shared<ngraph::opset6::Reshape>(first_dim_reduce, f_dim_shape, false);
        auto s_dim_red_axis = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0});
        auto second_dim_reduce = std::make_shared<ngraph::opset6::ReduceProd>(split->output(1),
                                                                              s_dim_red_axis);
        auto s_dim_shape = ngraph::opset6::Constant::create(ngraph::element::i64, {1}, {1});
        auto second_dim = std::make_shared<ngraph::opset6::Reshape>(second_dim_reduce, s_dim_shape, false);
        auto out_shape = std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{first_dim, second_dim},
                                                                  0);
        auto x_reshaped = std::make_shared<ngraph::opset6::Reshape>(x, out_shape, false);
        return {std::make_shared<ngraph::opset6::MatMul>(x_reshaped, y)};
    }
    return {std::make_shared<ngraph::opset6::MatMul>(x, y)};

}

}}}}