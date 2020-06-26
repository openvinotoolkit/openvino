// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/depth_to_space_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

bool check_block_first(const ngraph::Shape& shape_input, const ngraph::Shape& shape_reshape_before,
                       const ngraph::AxisVector& permutation, const ngraph::Shape& shape_reshape_after,
                       size_t& possible_block_size) {
    bool is_transformation_valid = true;
    uint64_t spatial_dims = shape_input.size() - 2;
    possible_block_size = shape_reshape_before[1];
    if (possible_block_size == 0)
        return false;
    uint64_t c_dim = shape_input[1] / std::pow(possible_block_size, spatial_dims);

    // x' = reshape(data, [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK])
    ngraph::Shape expected_shape = {shape_input[0]};
    for (uint64_t i = 0; i < spatial_dims; ++i)
        expected_shape.push_back(possible_block_size);
    expected_shape.push_back(c_dim);
    for (uint64_t i = 2; i < shape_input.size(); ++i)
        expected_shape.push_back(shape_input[i]);
    is_transformation_valid &= (expected_shape == shape_reshape_before);

    // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K])
    ngraph::AxisVector expected_permutation = {0, static_cast<size_t>(spatial_dims + 1)};
    for (uint64_t i = 2; i < shape_input.size(); ++i) {
        expected_permutation.push_back(spatial_dims + i);
        expected_permutation.push_back(i - 1);
    }
    is_transformation_valid &= (expected_permutation == permutation);

    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size, ..., DK * block_size])
    expected_shape = {shape_input[0], static_cast<size_t>(c_dim)};
    for (uint64_t i = 2; i < shape_input.size(); ++i)
        expected_shape.push_back(shape_input[i] * possible_block_size);
    is_transformation_valid &= (expected_shape == shape_reshape_after);

    return is_transformation_valid;
}

bool check_depth_first(const ngraph::Shape& shape_input, const ngraph::Shape& shape_reshape_before,
                       const ngraph::AxisVector& permutation, const ngraph::Shape& shape_reshape_after,
                       size_t& possible_block_size) {
    bool is_transformation_valid = true;
    uint64_t spatial_dims = shape_input.size() - 2;
    possible_block_size = shape_reshape_before[2];
    if (possible_block_size == 0)
        return false;
    uint64_t c_dim = shape_input[1] / std::pow(possible_block_size, spatial_dims);

    // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK])
    ngraph::Shape expected_shape = {shape_input[0], static_cast<size_t>(c_dim)};
    for (uint64_t i = 0; i < spatial_dims; ++i)
        expected_shape.push_back(possible_block_size);
    for (uint64_t i = 2; i < shape_input.size(); ++i)
        expected_shape.push_back(shape_input[i]);
    is_transformation_valid &= (expected_shape == shape_reshape_before);

    // x'' = transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
    ngraph::AxisVector expected_permutation = {0, 1};
    for (uint64_t i = 2; i < shape_input.size(); ++i) {
        expected_permutation.push_back(spatial_dims + i);
        expected_permutation.push_back(i);
    }
    is_transformation_valid &= (expected_permutation == permutation);

    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size, ..., DK * block_size])
    expected_shape = {shape_input[0], static_cast<size_t>(c_dim)};
    for (uint64_t i = 2; i < shape_input.size(); ++i)
        expected_shape.push_back(shape_input[i] * possible_block_size);
    is_transformation_valid &= (expected_shape == shape_reshape_after);

    return is_transformation_valid;
}

void ngraph::pass::DepthToSpaceFusion::depth_to_space_fusion() {
    auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto input2 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto input3 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto reshape_before = std::make_shared<ngraph::opset3::Reshape> (input0, input1, false);
    auto permute = std::make_shared<ngraph::opset3::Transpose> (reshape_before, input2);
    auto reshape_after = std::make_shared<ngraph::opset3::Reshape> (permute, input3, false);

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto reshape_after = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(m.get_match_root());
        if (!reshape_after) {
            return false;
        }

        auto permute = std::dynamic_pointer_cast<ngraph::opset3::Transpose>(reshape_after->input_value(0).get_node_shared_ptr());
        if (!permute || permute->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        auto reshape_before = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(permute->input_value(0).get_node_shared_ptr());
        if (!reshape_before || reshape_before->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        auto p_shape_input = reshape_before->get_input_partial_shape(0);
        auto p_shape_reshape_before = reshape_before->get_output_partial_shape(0);
        auto p_shape_permute = permute->get_output_partial_shape(0);
        auto p_shape_reshape_after = reshape_after->get_output_partial_shape(0);

        if (p_shape_input.is_dynamic() || p_shape_reshape_before.is_dynamic() ||
            p_shape_permute.is_dynamic() || p_shape_reshape_after.is_dynamic()) {
            return false;
        }

        auto shape_input = p_shape_input.get_shape();
        auto shape_reshape_before = p_shape_reshape_before.get_shape();
        auto shape_permute = p_shape_permute.get_shape();
        auto shape_reshape_after = p_shape_reshape_after.get_shape();

        if (shape_input.size() < 3) {
            return false;
        }

        // input shape: [ batch, C, spatial_dims], expected_shape = spatial_dims.size() * 2 + 2
        size_t expected_shape_size = (shape_input.size() - 2) * 2 + 2;
        if (shape_input.size() != shape_reshape_after.size() || shape_reshape_before.size() != expected_shape_size ||
            shape_permute.size() != expected_shape_size) {
            return false;
        }

        ngraph::AxisVector permutation;
        if (auto input_const = std::dynamic_pointer_cast<opset3::Constant>(permute->input_value(1).get_node_shared_ptr())) {
            permutation = input_const->get_axis_vector_val();
        } else {
            return false;
        }

        ngraph::opset3::DepthToSpace::DepthToSpaceMode mode;
        size_t block_size;
        if (check_depth_first(shape_input, shape_reshape_before, permutation, shape_reshape_after, block_size)) {
            mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
        } else if (check_block_first(shape_input, shape_reshape_before, permutation, shape_reshape_after, block_size)) {
            mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
        } else {
            return false;
        }

        auto depth_to_space =
                std::make_shared<ngraph::opset3::DepthToSpace>(reshape_before->input_value(0), mode, block_size);
        depth_to_space->set_friendly_name(reshape_after->get_friendly_name());
        ngraph::copy_runtime_info({reshape_before, permute, reshape_after}, depth_to_space);

        if (!m_transformation_callback(depth_to_space)) {
            return false;
        }

        ngraph::replace_node(reshape_after, depth_to_space);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_after, "DepthToSpaceFusion");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}