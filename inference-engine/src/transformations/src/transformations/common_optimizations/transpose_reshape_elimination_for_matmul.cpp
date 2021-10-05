// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>
#include "itt.hpp"

bool check_transposes(const std::vector<int64_t>& before_order, const std::vector<int64_t>& after_order, const int rank, const bool transposed_b) {
    if (before_order.size() != after_order.size())
        return false;

    if (transposed_b) {
        // order must be before : 0, 1, 2, ..., N-1, N-2
        std::vector<int64_t> start_order(rank);
        std::iota(start_order.begin(), start_order.begin() + rank - 2, 0);
        start_order[rank - 1] = rank - 2;
        start_order[rank - 2] = rank - 1;

        if (before_order != start_order)
            return false;

        // order must be after : N-2, ..., 1, 0, N-1
        std::vector<int64_t> back_order(rank);
        std::iota(back_order.begin() + 1, back_order.begin() + rank - 1, 0);
        back_order[0] = rank - 2;
        back_order[rank - 1] = rank - 1;

        if (after_order != back_order)
            return false;
    } else {
        // order must be before : N-2, N-1, 0, 1, 2, ...
        std::vector<int64_t> needed_transpose_order_before(rank);
        std::iota(needed_transpose_order_before.begin() + 2, needed_transpose_order_before.end(), 0);
        needed_transpose_order_before[0] = rank - 2;
        needed_transpose_order_before[1] = rank - 1;

        if (before_order != needed_transpose_order_before)
            return false;

        // order of transpose after matmul must be back for transpose before
        std::vector<int64_t> back_order(rank);
        for (auto i = 0; i < rank; i++)
            back_order[i] = std::distance(after_order.begin(), std::find(after_order.begin(), after_order.end(), i));

        if (before_order != back_order)
            return false;
    }

    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeReshapeEliminationForMatmul, "TransposeReshapeEliminationForMatmul", 0);

ngraph::pass::TransposeReshapeEliminationForMatmul::TransposeReshapeEliminationForMatmul() {
    MATCHER_SCOPE(TransposeReshapeEliminationForMatmul);
    auto input_1_pattern = ngraph::pattern::any_input();
    auto input_2_pattern = ngraph::pattern::any_input();

    auto const_transpose_before_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto transpose_before_pattern = ngraph::pattern::wrap_type<opset1::Transpose>({input_2_pattern, const_transpose_before_pattern});

    auto const_reshape_before_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto reshape_before_pattern = ngraph::pattern::wrap_type<opset1::Reshape>({transpose_before_pattern, const_reshape_before_pattern});

    auto matmul_pattern = ngraph::pattern::wrap_type<opset1::MatMul>({input_1_pattern, reshape_before_pattern});

    auto const_reshape_after_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto reshape_after_pattern = ngraph::pattern::wrap_type<opset1::Reshape>({matmul_pattern, const_reshape_after_pattern});

    auto const_transpose_after_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto transpose_after_pattern = ngraph::pattern::wrap_type<opset1::Transpose>({reshape_after_pattern, const_transpose_after_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input_1 = pattern_value_map.at(input_1_pattern);
        const auto& input_2 = pattern_value_map.at(input_2_pattern);

        auto matmul = std::dynamic_pointer_cast<opset1::MatMul>(pattern_value_map.at(matmul_pattern).get_node_shared_ptr());
        const bool transposed_a = matmul->get_transpose_a();
        const bool transposed_b = matmul->get_transpose_b();

        // check transpose order before and after matmul
        auto transpose_before = std::dynamic_pointer_cast<opset1::Transpose>(pattern_value_map.at(transpose_before_pattern).get_node_shared_ptr());
        auto transpose_after = std::dynamic_pointer_cast<opset1::Transpose>(pattern_value_map.at(transpose_after_pattern).get_node_shared_ptr());
        auto transpose_before_order =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_before->get_input_node_shared_ptr(1))->cast_vector<int64_t>();
        auto transpose_after_order =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_after->get_input_node_shared_ptr(1))->cast_vector<int64_t>();
        if (!check_transposes(transpose_before_order, transpose_after_order, transpose_before->get_input_shape(0).size(), transposed_b))
            return false;

        auto reshape_before = std::dynamic_pointer_cast<opset1::Reshape>(pattern_value_map.at(reshape_before_pattern).get_node_shared_ptr());
        auto reshape_after = std::dynamic_pointer_cast<opset1::Reshape>(pattern_value_map.at(reshape_after_pattern).get_node_shared_ptr());

        const auto new_matmul = std::make_shared<opset1::MatMul>(input_1, input_2, transposed_a, false);
        new_matmul->set_friendly_name(matmul->get_friendly_name());
        copy_runtime_info({transpose_before, reshape_before, matmul, reshape_after, transpose_after}, new_matmul);
        replace_node(transpose_after, new_matmul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_after_pattern, matcher_name);
    this->register_matcher(m, callback);
}
