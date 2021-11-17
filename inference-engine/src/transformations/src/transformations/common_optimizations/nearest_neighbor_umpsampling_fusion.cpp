// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

namespace {
std::vector<size_t> get_scales_from_mul_const_shape(const ngraph::Shape& s, uint64_t input_rank) {
    if (input_rank < 4 || static_cast<uint64_t>(s.size()) != 2 + 2 * (input_rank - 2)) return {};

    ngraph::Shape expected_shape(2 + 2 * (input_rank - 2), static_cast<size_t>(1));
    std::vector<size_t> scales(input_rank - 2);
    for (uint64_t i = 1; i <= input_rank - 2; ++i) {
        expected_shape[2 * i] = s[2 * i];
        scales[i - 1] = s[2 * i];
    }

    if (s != expected_shape) return {};

    return scales;
}

std::shared_ptr<ngraph::opset8::Unsqueeze> get_input_unsqueeze_for_concat_1(const std::shared_ptr<ngraph::opset8::Concat>& concat, const Shape& shape) {
    size_t rank = shape.size();

    const auto inputs = concat->input_values();
    size_t num_of_input_values = inputs.size();

    if (num_of_input_values != 2 + 2 * (rank - 2)) return nullptr;

    const auto input0 = std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(inputs[0].get_node_shared_ptr());
    if (!input0) return nullptr;

    const auto input0_axis = std::dynamic_pointer_cast<ngraph::opset8::Constant>(input0->input_value(1).get_node_shared_ptr());
    if (!input0_axis || input0_axis->cast_vector<int64_t>() != std::vector<int64_t>{0}) return nullptr;

    std::vector<int64_t> input_constants(num_of_input_values, 1);

    for (size_t i = 1; i < num_of_input_values; ++i) {
        const auto& current_input = std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(inputs[i].get_node_shared_ptr());
        if (!current_input) return nullptr;

        const auto current_input_axis = std::dynamic_pointer_cast<ngraph::opset8::Constant>(current_input->input_value(1).get_node_shared_ptr());
        if (!current_input_axis || current_input_axis->cast_vector<int64_t>() != std::vector<int64_t>{0}) return nullptr;

        const auto unsqueezed_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(current_input->input_value(0).get_node_shared_ptr());
        if (!unsqueezed_const) return nullptr;

        const auto unsqueezed_const_value = unsqueezed_const->cast_vector<int64_t>();
        if (unsqueezed_const_value.size() != 1) return nullptr;

        input_constants[i] = unsqueezed_const_value[0];
    }

    std::vector<int64_t> expected_input_constants(num_of_input_values, 1);
    for (size_t i = 1; i <= rank - 2; ++i) {
        expected_input_constants[2 * (i - 1) + 1] = static_cast<int64_t>(shape[i]);
    }
    expected_input_constants.back() = static_cast<int64_t>(shape.back());

    if (input_constants != expected_input_constants) return nullptr;

    return input0;
}
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::NearestNeighborUpsamplingFusion, "NearestNeighborUpsamplingFusion", 0);

ngraph::pass::NearestNeighborUpsamplingFusion::NearestNeighborUpsamplingFusion() {
    MATCHER_SCOPE(NearestNeighborUpsamplingFusion);
    auto input = ngraph::pattern::any_input();
    auto concat_1 = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
    auto concat_2 = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
    auto reshape_1 = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>({input, concat_1});
    auto mul_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul = pattern::wrap_type<ngraph::opset8::Multiply>({reshape_1, mul_const});
    auto reshape_2 = pattern::wrap_type<ngraph::opset8::Reshape>({mul, concat_2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto &pattern_to_output = m.get_pattern_value_map();

        const auto reshape_2_node = std::dynamic_pointer_cast<ngraph::opset8::Reshape>(pattern_to_output.at(reshape_2).get_node_shared_ptr());
        const auto mul_node = std::dynamic_pointer_cast<ngraph::opset8::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr());

        if (!reshape_2_node || !mul_node || mul_node->get_input_partial_shape(1).is_dynamic()) return false;

        const auto mul_const_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(mul_node->input_value(1).get_node_shared_ptr());
        if (!mul_const_node) return false;

        const auto reshape_1_node = std::dynamic_pointer_cast<ngraph::opset8::Reshape>(pattern_to_output.at(reshape_1).get_node_shared_ptr());
        if (!reshape_1_node || reshape_1_node->get_input_partial_shape(0).is_dynamic()) return false;

        uint64_t input_rank = static_cast<uint64_t>(reshape_1_node->get_input_partial_shape(0).rank().get_length());
        const auto mul_const_shape = mul_const_node->get_output_shape(0);
        const auto scales = get_scales_from_mul_const_shape(mul_const_shape, input_rank);
        if (scales.empty() || std::all_of(scales.begin(), scales.end(), [](size_t s) { return s == 1;})) { return false; }

        const auto mul_const_value = mul_const_node->cast_vector<float>();
        if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [](float x){ return x != 1.0f; })) { return false; }

        const auto concat_1_node = std::dynamic_pointer_cast<ngraph::opset8::Concat>(reshape_1_node->input_value(1).get_node_shared_ptr());
        if (!concat_1_node) return false;

        const auto unsqueeze_1 = get_input_unsqueeze_for_concat_1(concat_1_node, reshape_1_node->get_input_shape(0));
        if (!unsqueeze_1) return false;

        const auto concat_2_node = std::dynamic_pointer_cast<ngraph::opset8::Concat>(pattern_to_output.at(concat_2).get_node_shared_ptr());
        if (!concat_2_node) return false;

        std::shared_ptr<ngraph::opset8::Unsqueeze> unsqueeze_2;
        std::vector<int64_t> new_spatial_shape;
        std::tie(unsqueeze_2, new_spatial_shape) = get_input_unsqueeze_for_concat_2(concat_2_node, reshape_1_node->get_input_shape(0));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_2, matcher_name);
    register_matcher(m, callback);
}
