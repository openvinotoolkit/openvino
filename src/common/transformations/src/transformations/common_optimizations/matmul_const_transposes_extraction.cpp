// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
ov::pass::MatMulConstTransposesExtraction::MatMulConstTransposesExtraction() {
    auto data_pattern = ov::pass::pattern::any_input();
    auto weights_pattern = wrap_type<v0::Constant, v0::FakeQuantize>([](Output<Node> node) -> bool {
        const auto& pshape = node.get_partial_shape();
        const auto& rank = pshape.rank();
        return rank.is_static() && rank.get_length() >= 2 &&
               std::count(pshape.begin(), pshape.end(), 1) >= rank.get_length() - 2;
    });
    auto matmul_pattern = wrap_type<v0::MatMul>({data_pattern, weights_pattern});
    matcher_pass_callback callback = [=](Matcher& m) {
        auto node = m.get_match_root();
        auto matmul = as_type<v0::MatMul>(node.get());
        if (!matmul || matmul->get_transpose_b())
            return false;

        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& weights = pattern_value_map.at(weights_pattern);

        std::vector<int> transpose_order(weights.get_partial_shape().size());
        std::iota(transpose_order.begin(), transpose_order.end(), 0);
        std::reverse(transpose_order.end() - 2, transpose_order.end());
        std::shared_ptr<Node> transpose = std::make_shared<ov::op::v1::Transpose>(
            weights,
            v0::Constant::create(element::i32, {transpose_order.size()}, transpose_order));
        auto new_matmul = std::make_shared<v0::MatMul>(pattern_value_map.at(data_pattern),
                                                       transpose,
                                                       matmul->get_transpose_a(),
                                                       true);
        new_matmul->set_friendly_name(matmul->get_friendly_name());
        copy_runtime_info(node, {transpose, new_matmul});
        replace_node(node, new_matmul);
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_pattern, "MatMulConstTransposesExtraction");
    this->register_matcher(m, callback);
}
