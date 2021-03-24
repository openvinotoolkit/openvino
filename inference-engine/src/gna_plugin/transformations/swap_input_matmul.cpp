// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swap_input_matmul.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>


NGRAPH_RTTI_DEFINITION(ngraph::pass::SwapInputMatMul, "SwapInputMatMul", 0);

ngraph::pass::SwapInputMatMul::SwapInputMatMul() {
    auto matmul = pattern::wrap_type<opset1::MatMul>({pattern::any_input(pattern::has_static_shape()),
                                                      pattern::any_input(pattern::has_static_shape())},
                                                     pattern::has_static_shape());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(m.get_match_root());
        if (!matmul) {
            return false;
        }

        auto input_a = matmul->input(0).get_source_output();
        auto input_b = matmul->input(1).get_source_output();

        auto create_transpose = [this](Output<Node> node, const std::string& transpose_name) -> std::shared_ptr<Node> {
            Shape output_shape = node.get_node_shared_ptr()->get_shape();

            std::vector<size_t> transpose_order(output_shape.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose = register_new_node<ngraph::opset1::Transpose>(
                    node, opset1::Constant::create(element::i64, Shape {transpose_order.size()}, transpose_order));
            transpose->set_friendly_name(transpose_name);
            return transpose;
        };

        NodeVector new_ops;

        if (std::dynamic_pointer_cast<opset1::Constant>(input_a.get_node_shared_ptr())  ||
         std::dynamic_pointer_cast<opset1::FakeQuantize>(input_a.get_node_shared_ptr())) {
            auto reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2},
                    std::vector<size_t>{input_b.get_node_shared_ptr()->get_shape()[1],
                                        input_b.get_node_shared_ptr()->get_shape()[0]});
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(input_b, reshape_pattern, false);
            auto new_matmul = std::make_shared<ngraph::op::MatMul>(reshape, input_a, matmul->get_transpose_b(), !matmul->get_transpose_a());
            new_matmul->set_friendly_name(matmul->get_friendly_name());
            new_ops.push_back(new_matmul);

            auto transpose_out = create_transpose(new_matmul, new_matmul->get_friendly_name() + "/transpose_out");
            new_ops.push_back(transpose_out);

            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, transpose_out);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "SwapInputMatMul");
    this->register_matcher(m, callback);
}
