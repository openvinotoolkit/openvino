// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/depth_to_space_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

namespace {
bool check_block_first(const ngraph::PartialShape& shape_input,
                       const ngraph::PartialShape& shape_reshape_before,
                       const ngraph::AxisVector& permutation,
                       const ngraph::PartialShape& shape_reshape_after,
                       size_t& possible_block_size) {
    const auto input_rank = shape_input.rank();
    auto spatial_dims = input_rank.get_length() - 2;
    if (shape_reshape_before[1].is_dynamic() || shape_reshape_before[1].get_length() == 0) {
        return false;
    }

    possible_block_size = shape_reshape_before[1].get_length();
    ngraph::Dimension c_dim(
        static_cast<int64_t>(shape_input[1].get_length() / std::pow(possible_block_size, spatial_dims)));

    // x' = reshape(data, [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK])
    ngraph::PartialShape expected_shape = {shape_input[0]};
    for (int i = 0; i < spatial_dims; ++i)
        expected_shape.push_back(possible_block_size);
    expected_shape.push_back(c_dim);
    for (int i = 2; i < input_rank.get_length(); ++i)
        expected_shape.push_back(shape_input[i]);

    if (!ngraph::op::util::shapes_equal_except_dynamic_expected_batch(expected_shape, shape_reshape_before)) {
        return false;
    }

    // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K])
    ngraph::AxisVector expected_permutation = {0, static_cast<size_t>(spatial_dims + 1)};
    for (int i = 2; i < input_rank.get_length(); ++i) {
        expected_permutation.push_back(spatial_dims + i);
        expected_permutation.push_back(i - 1);
    }

    if (expected_permutation != permutation) {
        return false;
    }

    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size, ..., DK *
    // block_size])
    expected_shape = {shape_input[0], c_dim};
    for (int i = 2; i < input_rank.get_length(); ++i)
        expected_shape.push_back(shape_input[i] * possible_block_size);

    if (!ngraph::op::util::shapes_equal_except_dynamic_expected_batch(expected_shape, shape_reshape_after)) {
        return false;
    }

    return true;
}

bool check_depth_first(const ngraph::PartialShape& shape_input,
                       const ngraph::PartialShape& shape_reshape_before,
                       const ngraph::AxisVector& permutation,
                       const ngraph::PartialShape& shape_reshape_after,
                       size_t& possible_block_size) {
    const auto input_rank = shape_input.rank();
    auto spatial_dims = input_rank.get_length() - 2;
    if (shape_reshape_before[2].is_dynamic() || shape_reshape_before[2].get_length() == 0) {
        return false;
    }

    possible_block_size = shape_reshape_before[2].get_length();
    ngraph::Dimension c_dim(
        static_cast<int>(shape_input[1].get_length() / std::pow(possible_block_size, spatial_dims)));

    // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK])
    ngraph::PartialShape expected_shape = {shape_input[0], c_dim};
    for (int i = 0; i < spatial_dims; ++i)
        expected_shape.push_back(possible_block_size);
    for (int i = 2; i < input_rank.get_length(); ++i)
        expected_shape.push_back(shape_input[i]);

    if (!ngraph::op::util::shapes_equal_except_dynamic_expected_batch(expected_shape, shape_reshape_before)) {
        return false;
    }

    // x'' = transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
    ngraph::AxisVector expected_permutation = {0, 1};
    for (int i = 2; i < input_rank.get_length(); ++i) {
        expected_permutation.push_back(spatial_dims + i);
        expected_permutation.push_back(i);
    }
    if (expected_permutation != permutation) {
        return false;
    }

    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size, ..., DK *
    // block_size])
    expected_shape = {shape_input[0], c_dim};
    for (int i = 2; i < input_rank.get_length(); ++i)
        expected_shape.push_back(shape_input[i] * possible_block_size);

    if (!ngraph::op::util::shapes_equal_except_dynamic_expected_batch(expected_shape, shape_reshape_after)) {
        return false;
    }

    return true;
}

}  // namespace

ngraph::pass::DepthToSpaceFusion::DepthToSpaceFusion() {
    MATCHER_SCOPE(DepthToSpaceFusion);
    auto input0 = ngraph::pattern::any_input(pattern::rank_equals(4));
    auto input1 = ngraph::pattern::any_input();
    auto input2 = ngraph::pattern::any_input();
    auto input3 = ngraph::pattern::any_input();
    auto reshape_before =
        ngraph::pattern::wrap_type<ngraph::opset3::Reshape>({input0, input1}, pattern::consumers_count(1));
    auto permute =
        ngraph::pattern::wrap_type<ngraph::opset3::Transpose>({reshape_before, input2}, pattern::consumers_count(1));
    auto reshape_after = ngraph::pattern::wrap_type<ngraph::opset3::Reshape>({permute, input3});

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto reshape_after = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(m.get_match_root());
        if (!reshape_after) {
            return false;
        }

        auto permute =
            std::dynamic_pointer_cast<ngraph::opset3::Transpose>(reshape_after->get_input_node_shared_ptr(0));
        if (!permute) {
            return false;
        }

        auto reshape_before = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(permute->get_input_node_shared_ptr(0));
        if (!reshape_before) {
            return false;
        }

        auto p_shape_input = reshape_before->get_input_partial_shape(0);
        auto p_shape_reshape_before = reshape_before->get_output_partial_shape(0);
        auto p_shape_permute = permute->get_output_partial_shape(0);
        auto p_shape_reshape_after = reshape_after->get_output_partial_shape(0);

        const auto input_rank = p_shape_input.rank();
        if (input_rank.is_dynamic() || p_shape_input.rank().get_length() < 3) {
            return false;
        }

        // check that all dimensions except batch are static
        if (std::any_of(p_shape_input.begin() + 1, p_shape_input.end(), [](const ngraph::Dimension& x) {
                return x.is_dynamic();
            })) {
            return false;
        }

        // input shape: [ batch, C, spatial_dims], expected_shape = spatial_dims.size() * 2 + 2
        auto expected_shape_size = (input_rank.get_length() - 2) * 2 + 2;
        if (input_rank != p_shape_reshape_after.rank().get_length() ||
            p_shape_reshape_before.rank().get_length() != expected_shape_size ||
            p_shape_permute.rank().get_length() != expected_shape_size) {
            return false;
        }

        ngraph::AxisVector permutation;
        if (auto input_const = std::dynamic_pointer_cast<opset3::Constant>(permute->get_input_node_shared_ptr(1))) {
            permutation = input_const->get_axis_vector_val();
        } else {
            return false;
        }

        ngraph::opset3::DepthToSpace::DepthToSpaceMode mode;
        size_t block_size;
        if (check_depth_first(p_shape_input, p_shape_reshape_before, permutation, p_shape_reshape_after, block_size)) {
            mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
        } else if (check_block_first(p_shape_input,
                                     p_shape_reshape_before,
                                     permutation,
                                     p_shape_reshape_after,
                                     block_size)) {
            mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
        } else {
            return false;
        }

        auto depth_to_space =
            std::make_shared<ngraph::opset3::DepthToSpace>(reshape_before->input_value(0), mode, block_size);
        depth_to_space->set_friendly_name(reshape_after->get_friendly_name());
        ngraph::copy_runtime_info({reshape_before, permute, reshape_after}, depth_to_space);
        ngraph::replace_node(reshape_after, depth_to_space);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_after, matcher_name);
    register_matcher(m, callback);
}
