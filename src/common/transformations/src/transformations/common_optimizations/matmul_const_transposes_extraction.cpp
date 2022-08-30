// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>

ngraph::pass::MatMulConstTransposesExtraction::MatMulConstTransposesExtraction() {
    auto data_pattern = pattern::any_input();
    auto weights_pattern = pattern::wrap_type<opset8::Constant, opset8::FakeQuantize>([](Output<Node> node) -> bool {
        const auto& pshape = node.get_partial_shape();
        const auto& rank = pshape.rank();
        return rank.is_static() && rank.get_length() >= 2 &&
               std::count(pshape.begin(), pshape.end(), 1) >= rank.get_length() - 2;
    });
    auto matmul_pattern = pattern::wrap_type<opset8::MatMul>({data_pattern, weights_pattern});
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto matmul = as_type<opset8::MatMul>(node.get());
        if (!matmul || matmul->get_transpose_b())
            return false;

        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& weights = pattern_value_map.at(weights_pattern);

        std::vector<int> transpose_order(weights.get_partial_shape().size());
        std::iota(transpose_order.begin(), transpose_order.end(), 0);
        std::reverse(transpose_order.end() - 2, transpose_order.end());
        std::shared_ptr<Node> transpose = std::make_shared<opset8::Transpose>(
            weights,
            op::Constant::create(element::i32, {transpose_order.size()}, transpose_order));
        if (ov::is_type<op::Constant>(weights.get_node())) {
            if (auto constant = get_constant_from_source(transpose))
                transpose = constant;
        }
        auto new_matmul = std::make_shared<opset8::MatMul>(pattern_value_map.at(data_pattern),
                                                           transpose,
                                                           matmul->get_transpose_a(),
                                                           true);
        new_matmul->set_friendly_name(matmul->get_friendly_name());
        copy_runtime_info(node, {transpose, new_matmul});
        replace_node(node, new_matmul);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(matmul_pattern, "MatMulConstTransposesExtraction");
    this->register_matcher(m, callback);
}
