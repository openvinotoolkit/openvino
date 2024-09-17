// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {
/// \brief      Check for correct Transpose orders which are before and after MatMul. Second Transpose must be back for
///  first Transpose before MatMul
///
/// \param      before_order       Order of Transpose which is before MatMul
/// \param      after_order        Order of Transpose which is after MatMul
/// \param      transposed_b       true - second MatMul input is transposed, otherwise, it's not transposed
///
/// \return     True - Transposes have right orders, otherwise, Transposes have incorrect order for transformation
///
bool check_transposes(const std::vector<int64_t>& before_order,
                      const std::vector<int64_t>& after_order,
                      const bool transposed_b) {
    const size_t rank = before_order.size();
    if (rank < 3)
        return false;

    if (before_order.size() != after_order.size())
        return false;

    if (transposed_b) {
        // before order must be : 0, 1, 2, ..., N-1, N-2
        std::vector<int64_t> start_order(rank);
        std::iota(start_order.begin(), start_order.begin() + rank - 2, 0);
        start_order[rank - 1] = rank - 2;
        start_order[rank - 2] = rank - 1;

        if (before_order != start_order)
            return false;

        // after order must be : 1, ..., N-2, 0, N-1
        std::vector<int64_t> back_order(rank);
        std::iota(back_order.begin(), back_order.begin() + rank - 2, 1);
        back_order[rank - 2] = 0;
        back_order[rank - 1] = rank - 1;

        if (after_order != back_order)
            return false;
    } else {
        // before order must be : N-2, N-1, 0, 1, 2, ...
        std::vector<int64_t> needed_transpose_order_before(rank);
        std::iota(needed_transpose_order_before.begin() + 2, needed_transpose_order_before.end(), 0);
        needed_transpose_order_before[0] = rank - 2;
        needed_transpose_order_before[1] = rank - 1;

        if (before_order != needed_transpose_order_before)
            return false;

        // transpose order after matmul must be back for transpose before
        std::vector<int64_t> back_order(rank);
        for (size_t i = 0; i < rank; i++)
            back_order[i] = std::distance(after_order.begin(), std::find(after_order.begin(), after_order.end(), i));

        if (before_order != back_order)
            return false;
    }

    return true;
}

/// \brief      Check for input Reshape which are before MatMul
///
/// \param      reshape            Reshape which is before MatMul
/// \param      new_shape          New shape for Reshape
/// \param      transposed_b       true - second MatMul input is transposed, otherwise, it's not transposed
///
/// \return     True - Reshape has right new shape for reshaping, otherwise, Reshape has incorrect new shape for
/// transformation
///
bool check_input_reshape(const std::shared_ptr<ov::op::v1::Reshape>& reshape,
                         const std::vector<int64_t>& new_shape,
                         const bool transposed_b) {
    const auto input_shape = reshape->get_input_shape(0);
    const size_t input_rank = input_shape.size();
    const size_t output_rank = reshape->get_output_shape(0).size();
    if (input_rank < 3 || output_rank != 2)
        return false;

    if (transposed_b) {
        const int64_t k = input_shape.back();
        const int64_t new_n = ov::shape_size(input_shape) / k;
        if (new_shape != std::vector<int64_t>{new_n, k})
            return false;
    } else {
        const int64_t k = input_shape.front();
        const int64_t new_n = ov::shape_size(input_shape) / k;
        if (new_shape != std::vector<int64_t>{k, -1} && new_shape != std::vector<int64_t>{k, new_n})
            return false;
    }

    return true;
}
}  // namespace

ov::pass::TransposeReshapeEliminationForMatmul::TransposeReshapeEliminationForMatmul() {
    MATCHER_SCOPE(TransposeReshapeEliminationForMatmul);
    auto input_1_pattern = pass::pattern::any_input([](const Output<Node>& node) -> bool {
        const auto& shape = node.get_partial_shape();
        const auto& rank = shape.rank();
        return rank.is_static() && rank.get_length() == 2 && shape.is_static();
    });
    auto input_2_pattern = pass::pattern::any_input([](const Output<Node>& node) -> bool {
        return node.get_partial_shape().is_static();
    });

    auto const_transpose_before_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_before_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({input_2_pattern, const_transpose_before_pattern});

    auto const_reshape_before_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reshape_before_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({transpose_before_pattern, const_reshape_before_pattern});

    auto matmul_pattern = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({input_1_pattern, reshape_before_pattern});

    auto const_reshape_after_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reshape_after_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({matmul_pattern, const_reshape_after_pattern});

    auto const_transpose_after_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_after_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({reshape_after_pattern, const_transpose_after_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input_1 = pattern_value_map.at(input_1_pattern);
        const auto& input_2 = pattern_value_map.at(input_2_pattern);

        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_value_map.at(matmul_pattern).get_node_shared_ptr());
        if (!matmul)
            return false;
        const bool transposed_a = matmul->get_transpose_a();
        const bool transposed_b = matmul->get_transpose_b();

        auto reshape_before =
            ov::as_type_ptr<ov::op::v1::Reshape>(pattern_value_map.at(reshape_before_pattern).get_node_shared_ptr());
        auto reshape_after =
            ov::as_type_ptr<ov::op::v1::Reshape>(pattern_value_map.at(reshape_after_pattern).get_node_shared_ptr());
        auto reshape_before_constant = ov::as_type_ptr<ov::op::v0::Constant>(
            pattern_value_map.at(const_reshape_before_pattern).get_node_shared_ptr());
        if (!reshape_before || !reshape_after || !reshape_before_constant)
            return false;
        if (!check_input_reshape(reshape_before, reshape_before_constant->cast_vector<int64_t>(), transposed_b))
            return false;

        // check transpose order before and after matmul
        auto transpose_before = ov::as_type_ptr<ov::op::v1::Transpose>(
            pattern_value_map.at(transpose_before_pattern).get_node_shared_ptr());
        auto transpose_after =
            ov::as_type_ptr<ov::op::v1::Transpose>(pattern_value_map.at(transpose_after_pattern).get_node_shared_ptr());
        if (!transpose_before || !transpose_after)
            return false;

        auto transpose_before_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(transpose_before->get_input_node_shared_ptr(1));
        auto transpose_after_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(transpose_after->get_input_node_shared_ptr(1));
        if (!transpose_before_constant || !transpose_after_constant)
            return false;

        auto transpose_before_order = transpose_before_constant->cast_vector<int64_t>();
        auto transpose_after_order = transpose_after_constant->cast_vector<int64_t>();
        // need to check that input shape is correctly contracted and output shape is correctly unpacked using
        // transposes
        if (!check_transposes(transpose_before_order, transpose_after_order, transposed_b))
            return false;

        const auto new_matmul = std::make_shared<ov::op::v0::MatMul>(input_1, input_2, transposed_a, false);
        new_matmul->set_friendly_name(transpose_after->get_friendly_name());
        copy_runtime_info({transpose_before, reshape_before, matmul, reshape_after, transpose_after}, new_matmul);
        replace_node(transpose_after, new_matmul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_after_pattern, matcher_name);
    this->register_matcher(m, callback);
}
