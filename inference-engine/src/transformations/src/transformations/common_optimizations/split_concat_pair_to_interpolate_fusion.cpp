// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp"

#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace {
template<typename T>
std::vector<std::vector<T>> grouped_vector(const std::vector<T>& v) {
    std::vector<std::vector<T>> result;

    if (v.empty()) return std::vector<std::vector<T>>{};

    T prev = v[0];
    std::vector<T> group = {prev};
    size_t num_of_elems = v.size();

    for (size_t i = 1; i < num_of_elems; ++i) {
        T x{v[i]};
        if (prev == x) {
            group.emplace_back(x);
            prev = x;
        } else {
            result.emplace_back(group);
            prev = x;
            group = {prev};
        }
    }
    result.emplace_back(group);
    return result;
}

std::shared_ptr<ngraph::opset8::Concat> get_concat_after_split(const std::shared_ptr<ngraph::opset8::Split>& split) {
    // This function gets consumers of the 'split' node, checks that the following conditions are fulfilled:
    // 1) 'split' node has only one consumer;
    // 2) for any output port of 'split', number of corresponding input ports of the consumer is the same;
    // 3) for any output port 'i' of the 'split', corresponding input ports of the consumer are
    //    [i * m, ..., i * m + (m - 1)], where 'm' is the same for all 'i';
    // 4) the consumer operation is 'Concat';
    // 5) 'split' is a unique producer for this 'Concat';
    // and, if all these conditions are fulfilled, returns the above mentioned 'Concat' node. Otherwise, if some of these
    // conditions is false, this functions returns nullptr.

    // If number of output nodes of 'split' is not equal to 1, then the transformation is not applicable.
    // Also, the transformation is not applicable, if there are some non-Concat consumers of 'split'.
    std::unordered_set<std::shared_ptr<ngraph::opset8::Concat>> concats;
    for (auto output : split->outputs()) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset8::Concat>(output.get_node_shared_ptr());
        if (!concat) return nullptr;
        concats.insert(concat);
    }
    if (concats.size() != 1) return nullptr;

    auto concat = *(concats.begin());

    // The transformation is applicable, only if 'split' is a unique producer for Concat.
    std::vector<size_t> idx;
    for (auto input : concat->input_values()) {
        if (input.get_node() != split.get()) return nullptr;
        idx.emplace_back(input.get_index());
    }

    // If numbers of consumer ports are various for various output ports of 'split', then the transformation is not applicable.
    auto grouped_idx = grouped_vector(idx);
    std::set<size_t> sizes_of_groups;
    for (const auto& group : grouped_idx) {
        sizes_of_groups.insert(group.size());
    }
    if (sizes_of_groups.size() != 1) return nullptr;

    // The transformation is applicable iff output port 0 of 'split' goes to ports [0, ..., m-1] of next node,
    // output port 1 of 'split' goes to ports [m, ..., m + (m-1)] of next node, ..., output port i of 'split'
    // goes to ports [i * m, ..., i * m + (m - 1)], and so on.
    std::vector<size_t> expected_ports_consuming_split(idx.size());
    std::iota(expected_ports_consuming_split.begin(), expected_ports_consuming_split.end(), 0);
    if (idx != expected_ports_consuming_split) return nullptr;

    return concat;
}

int64_t get_split_scale(const std::shared_ptr<ngraph::opset8::Split>& split_node,
                        const std::shared_ptr<ngraph::opset8::Concat>& concat) {
    // The transformation is applicable, only if the number of output ports of 'split' is multiple of
    // the number of inputs of Concat.
    // This function returns 0, if the number of output ports of 'split' is not multiple of the number of inputs of Concat,
    // and number_of_concat_inputs / number_of_output_ports_of_split otherwise
    const auto concat_inputs = concat->input_values();
    size_t num_of_concat_inputs = concat_inputs.size();

    std::unordered_set<size_t> split_output_ports;
    for (auto input : concat_inputs) {
        split_output_ports.insert(input.get_index());
    }
    size_t num_of_output_ports_of_split = split_output_ports.size();

    float float_scale = static_cast<float>(num_of_concat_inputs) / static_cast<float>(num_of_output_ports_of_split);
    size_t int_scale = num_of_concat_inputs / num_of_output_ports_of_split;

    if (float_scale != static_cast<float>(int_scale)) return 0;

    return static_cast<int64_t>(int_scale);
}
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::SplitConcatPairToInterpolateFusion, "SplitConcatPairToInterpolateFusion", 0);

ngraph::pass::SplitConcatPairToInterpolateFusion::SplitConcatPairToInterpolateFusion() {
    MATCHER_SCOPE(SplitConcatPairToInterpolateFusion);
    auto split_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Split>({pattern::any_input(pattern::has_static_shape()),
                                                                            pattern::any_input(pattern::has_static_shape())});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto split = std::dynamic_pointer_cast<ngraph::opset8::Split>(pattern_to_output.at(split_pattern).get_node_shared_ptr());
        if (!split) return false;

        Shape split_input_shape = split->get_input_shape(0);
        size_t split_input_rank = split_input_shape.size();
        if (split_input_rank != 4 && split_input_rank != 5) return false;

        auto concat = get_concat_after_split(split);

        if (!concat) return false;

        auto split_axis_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis_const) return false;

        int64_t axis = split_axis_const->cast_vector<int64_t>()[0];

        int64_t scale_factor = get_split_scale(split, concat);
        if (scale_factor == 0) return false;

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

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(split->input_value(0), cast_mul_result_to_int,
                                                                         scales_node, axis_node, attrs);

        interpolate->set_friendly_name(concat->get_friendly_name());
        ngraph::copy_runtime_info({split, concat},
                                  {interpolate, cast_mul_result_to_int, scales_node, axis_node, floor_node, mul_node,
                                   cast_shape_to_float, strided_slice_node, sslice_begin, sslice_end});
        ngraph::replace_node(concat, interpolate);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(split_pattern, matcher_name);
    register_matcher(m, callback);
}
