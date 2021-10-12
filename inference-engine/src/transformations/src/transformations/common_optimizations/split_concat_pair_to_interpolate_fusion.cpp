// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace {
// This function creates a partition of its argument into groups consisting of adjacent identical elements.
// Argument: std::vector<size_t> v
// Returns: partition of the argument
std::vector<std::vector<size_t>> grouped_vector(const std::vector<size_t>& v) {
    std::vector<std::vector<size_t>> result;

    if (v.empty()) return result;

    size_t prev = v[0];
    std::vector<size_t> group;

    for (const auto& x : v) {
        if (prev != x) {
            result.emplace_back(group);
            group.clear();
            prev = x;
        }
        group.emplace_back(x);
    }
    result.emplace_back(group);
    return result;
}

std::pair<std::shared_ptr<ngraph::opset8::Split>, int64_t> get_split_before_concat(const std::shared_ptr<ngraph::opset8::Concat>& concat) {
    // This function gets producers of the 'concat' node, checks that the following conditions are fulfilled:
    // 1) all producers for 'concat' are Split nodes;
    // 2) 'concat' has only one unique producer ('split');
    // 3) 'split' node has only one consumer;
    // 4) for any output port of 'split', number of corresponding input ports of the consumer is the same;
    // 5) for any output port 'i' of the 'split', corresponding input ports of the consumer are
    //    [i * m, ..., i * m + (m - 1)], where 'm' is the same for all 'i';
    // and, if all these conditions are fulfilled, returns the above mentioned 'Concat' node. Otherwise, if some of these
    // conditions is false, this functions returns nullptr.

    std::vector<size_t> idx;
    std::unordered_set<std::shared_ptr<ngraph::opset8::Split>> splits;
    for (const auto& input : concat->input_values()) {
        // If 'concat' has some non-Split producer, then the transformation is not applicable.
        auto split = std::dynamic_pointer_cast<ngraph::opset8::Split>(input.get_node_shared_ptr());
        if (!split) return {};
        idx.emplace_back(input.get_index());
        splits.insert(split);
    }
    // If 'concat' has more than one Splits as producers, then the transformation is not applicable.
    if (splits.size() != 1) return {};

    auto split = *(splits.begin());

    // If 'split' node has more than one consumer, then the transformation is not applicable.
    for (const auto& output : split->outputs()) {
        for (const auto& consumer : output.get_target_inputs()) {
            if (consumer.get_node() != concat.get()) return {};
        }
    }

    // If numbers of consumer ports are various for various output ports of 'split', then the transformation is not applicable.
    auto grouped_idx = grouped_vector(idx);
    std::unordered_set<size_t> sizes_of_groups;
    for (const auto& group : grouped_idx) {
        sizes_of_groups.insert(group.size());
    }
    if (sizes_of_groups.size() != 1) return {};
    int64_t size_of_group = static_cast<int64_t>(*(sizes_of_groups.begin()));

    // The transformation is applicable iff output port 0 of 'split' goes to ports [0, ..., m-1] of next node,
    // output port 1 of 'split' goes to ports [m, ..., m + (m-1)] of next node, ..., output port i of 'split'
    // goes to ports [i * m, ..., i * m + (m - 1)], and so on.
    for (size_t i = 0; i < grouped_idx.size(); ++i) {
        const auto& current_group = grouped_idx[i];
        if (std::any_of(current_group.begin(), current_group.end(), [i](size_t j){ return j != i; })) { return {}; }
    }

    return {split, size_of_group};
}
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::SplitConcatPairToInterpolateFusion, "SplitConcatPairToInterpolateFusion", 0);

ngraph::pass::SplitConcatPairToInterpolateFusion::SplitConcatPairToInterpolateFusion() {
    MATCHER_SCOPE(SplitConcatPairToInterpolateFusion);
    // Detect only concat, because we don't know how many inputs will go into concat.
    auto concat_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset8::Concat>(m.get_match_root());
        if (!concat) return false;

        int64_t scale_factor;
        std::shared_ptr<opset8::Split> split;
        std::tie(split, scale_factor) = get_split_before_concat(concat);
        if (!split || !scale_factor) return false;

        if (split->get_input_partial_shape(0).rank().is_dynamic()) return false;
        int64_t split_input_rank = split->get_input_partial_shape(0).rank().get_length();
        // If this transformation is applied in the case of the the rank is less than 4, we have a performance degradation.
        // And, at this time, we have no models with Split->Concat pattern when this transformation is applicable and
        // input rank of Split is greater than 5.
        if (split_input_rank != 4 && split_input_rank != 5) return false;

        auto split_axis_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis_const) return false;

        int64_t axis = split_axis_const->cast_vector<int64_t>()[0];

        ngraph::opset8::Interpolate::InterpolateAttrs attrs;

        attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
        attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(split->input_value(0));

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {1};
        std::vector<int64_t> end_mask = {1};
        std::vector<int64_t> new_axis_mask = {0};
        std::vector<int64_t> shrink_axis_mask = {0};
        std::vector<int64_t> ellipsis_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask,
                                                                                 end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = register_new_node<ngraph::opset8::Interpolate>(split->input_value(0), cast_mul_result_to_int,
                                                                          scales_node, axis_node, attrs);

        interpolate->set_friendly_name(concat->get_friendly_name());
        ngraph::copy_runtime_info(concat, {scales_node, axis_node, shape_node, sslice_begin, sslice_end, strided_slice_node, cast_shape_to_float, mul_node,
                                           floor_node, cast_mul_result_to_int, interpolate});
        ngraph::replace_node(concat, interpolate);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}
