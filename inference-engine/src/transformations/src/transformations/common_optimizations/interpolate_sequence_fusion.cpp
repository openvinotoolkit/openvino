// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/interpolate_sequence_fusion.hpp"

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
//#include <transformations/rt_info/disable_constant_folding.hpp>
//
//namespace {
//// This function creates a partition of its argument into groups consisting of adjacent identical elements.
//// Argument: std::vector<uint64_t> v
//// Returns: partition of the argument
//std::vector<std::vector<uint64_t>> grouped_vector(const std::vector<uint64_t>& v) {
//    std::vector<std::vector<uint64_t>> result;
//
//    if (v.empty()) return result;
//
//    uint64_t prev = v[0];
//    std::vector<uint64_t> group;
//
//    for (const auto& x : v) {
//        if (prev != x) {
//            result.emplace_back(group);
//            group.clear();
//            prev = x;
//        }
//        group.emplace_back(x);
//    }
//    result.emplace_back(group);
//    return result;
//}
//
//std::pair<std::shared_ptr<ngraph::opset8::Split>, uint64_t> get_split_before_concat(const std::shared_ptr<ngraph::opset8::Concat>& concat) {
//    // This function gets producers of the 'concat' node, checks that the following conditions are fulfilled:
//    // 1) all producers for 'concat' are Split nodes;
//    // 2) 'concat' has only one unique producer ('split');
//    // 3) 'split' node has only one consumer;
//    // 4) for any output port of 'split', number of corresponding input ports of the consumer is the same;
//    // 5) for any output port 'i' of the 'split', corresponding input ports of the consumer are
//    //    [i * m, ..., i * m + (m - 1)], where 'm' is the same for all 'i';
//    // and, if all these conditions are fulfilled, returns the above mentioned 'Concat' node. Otherwise, if some of these
//    // conditions is false, this functions returns nullptr.
//
//    std::vector<uint64_t> idx;
//    std::shared_ptr<ngraph::opset8::Split> split;
//    for (const auto& input : concat->input_values()) {
//        // If 'concat' has some non-Split producer, then the transformation is not applicable.
//        auto split_op = std::dynamic_pointer_cast<ngraph::opset8::Split>(input.get_node_shared_ptr());
//        if (!split)
//            split = split_op;
//        if (!split_op || split != split_op) return {};
//        idx.emplace_back(static_cast<uint64_t>(input.get_index()));
//    }
//
//    // If 'split' node has more than one consumer, then the transformation is not applicable.
//    for (const auto& output : split->outputs()) {
//        for (const auto& consumer : output.get_target_inputs()) {
//            if (consumer.get_node() != concat.get()) return {};
//        }
//    }
//
//    // If numbers of consumer ports are various for various output ports of 'split', then the transformation is not applicable.
//    auto grouped_idx = grouped_vector(idx);
//    std::unordered_set<uint64_t> sizes_of_groups;
//    for (const auto& group : grouped_idx) {
//        sizes_of_groups.insert(static_cast<uint64_t>(group.size()));
//    }
//    if (sizes_of_groups.size() != 1) return {};
//    uint64_t size_of_group = *(sizes_of_groups.begin());
//
//    // The transformation is applicable iff output port 0 of 'split' goes to ports [0, ..., m-1] of next node,
//    // output port 1 of 'split' goes to ports [m, ..., m + (m-1)] of next node, ..., output port i of 'split'
//    // goes to ports [i * m, ..., i * m + (m - 1)], and so on.
//    for (uint64_t i = 0; i < static_cast<uint64_t>(grouped_idx.size()); ++i) {
//        const auto& current_group = grouped_idx[i];
//        if (std::any_of(current_group.begin(), current_group.end(), [i](uint64_t j){ return j != i; })) { return {}; }
//    }
//
//    return {split, size_of_group};
//}
//} // namespace
//
//NGRAPH_RTTI_DEFINITION(ngraph::pass::SplitConcatPairToInterpolateFusion, "SplitConcatPairToInterpolateFusion", 0);
//
//ngraph::pass::SplitConcatPairToInterpolateFusion::SplitConcatPairToInterpolateFusion(bool use_shape_for_elimination) {
//    MATCHER_SCOPE(SplitConcatPairToInterpolateFusion);
//    // This transformation looks for Interpolate layer implemented using simple operations, namely Split and Concat,
//    // and replaces found pattern with a sequence of Shape, StridedSlice, Const, Mul, Interpolate.
//    // Found pattern:
//    //     Split -> Concat
//    // Here we assume that
//    //     1) input data of Split is 4D or 5D tensor;
//    //     2) split dimensions for 'split' belongs to {1, 2, 3};
//    //     3) all outputs of 'split' go to only inputs of 'concat';
//    //     4) 'concat' takes inputs only from 'split';
//    //     5) split_dim of 'split' is equal to axis of 'concat';
//    //     6) output port 0 of 'split' goes to ports [0, ..., m-1] of next node, output port 1 of 'split' goes to ports
//    //        [m, ..., m + (m-1)] of next node, ..., output port i of 'split' goes to ports [i * m, ..., i * m + (m - 1)],
//    //        and so on;
//    //     7) number of outputs of 'split' is equal to the length of the split axis.
//    // Such subgraph
//    //     Split -> Concat
//    // can be replaced with the Interpolate layer with the following attributes:
//    //     mode = nearest
//    //     shape_calculation_mode = scales
//    //     nearest_mode = round_prefer_floor
//    //     pads_begin = {0}
//    //     pads_end = {0}
//    //     antialias = false
//    //     coordinate_transformation_mode = half_pixel
//    //     cube_coeff = -0.75
//    // Next, the scaling factor in Interpolate is equal to a quotient of dividing number of input ports of 'concat'
//    // by number of output ports of 'split'.
//    //
//    // Detect only concat, because we don't know how many inputs will go into concat.
//    auto concat_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
//    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
//        using namespace ngraph;
//
//        auto concat = std::dynamic_pointer_cast<opset8::Concat>(m.get_match_root());
//        if (!concat) return false;
//
//        uint64_t scale_factor;
//        std::shared_ptr<opset8::Split> split;
//        std::tie(split, scale_factor) = get_split_before_concat(concat);
//        // If scale_factor == 1, then output data of Interpolate are equal to input data. Hence, we should not replace
//        // Split->Concat pair with Interpolate.
//        if (!split || !scale_factor || scale_factor == 1) return false;
//
//        if (split->get_input_partial_shape(0).rank().is_dynamic()) return false;
//        int64_t split_input_rank = split->get_input_partial_shape(0).rank().get_length();
//        // If this transformation is applied in the case of the the rank is less than 4, we have a performance degradation.
//        // And, at this time, we have no models with Split->Concat pattern when this transformation is applicable and
//        // input rank of Split is greater than 5.
//        if (split_input_rank != 4 && split_input_rank != 5) return false;
//
//        auto split_axis_const = std::dynamic_pointer_cast<opset8::Constant>(split->input_value(1).get_node_shared_ptr());
//        if (!split_axis_const) return false;
//
//        int64_t axis = split_axis_const->cast_vector<int64_t>()[0];
//
//        if (split->get_input_partial_shape(0).is_static() && split->get_input_shape(0)[axis] != split->outputs().size()) return false;
//
//        opset8::Interpolate::InterpolateAttrs attrs;
//
//        attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
//        attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
//        attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
//        attrs.pads_begin = std::vector<size_t>{0};
//        attrs.pads_end = std::vector<size_t>{0};
//        attrs.antialias = false;
//        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
//        attrs.cube_coeff = -0.75f;
//
//        auto scales_node = opset8::Constant::create(element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
//        auto axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{axis});
//        auto shape_node = std::make_shared<opset8::ShapeOf>(split->input_value(0));
//
//        auto sslice_begin = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{axis});
//        auto sslice_end = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{axis + 1});
//        std::vector<int64_t> begin_mask = {0};
//        std::vector<int64_t> end_mask = {0};
//        auto strided_slice_node = std::make_shared<opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);
//
//        auto cast_shape_to_float = std::make_shared<opset8::Convert>(strided_slice_node, element::f32);
//        auto mul_node = std::make_shared<opset8::Multiply>(cast_shape_to_float, scales_node);
//        auto floor_node = std::make_shared<opset8::Floor>(mul_node);
//        auto cast_mul_result_to_int = std::make_shared<opset8::Convert>(floor_node, element::i64);
//
//        std::shared_ptr<Node> sizes_node;
//
//        if (use_shape_for_elimination) {
//            sizes_node = get_constant_from_source(cast_mul_result_to_int);
//        } else {
//            disable_constant_folding(shape_node);
//        }
//
//        if (!sizes_node)
//            sizes_node = cast_mul_result_to_int;
//
//        auto interpolate = register_new_node<opset8::Interpolate>(split->input_value(0), sizes_node, scales_node, axis_node, attrs);
//
//        interpolate->set_friendly_name(concat->get_friendly_name());
//        copy_runtime_info(concat, {scales_node, axis_node, shape_node, sslice_begin, sslice_end, strided_slice_node, cast_shape_to_float, mul_node,
//                                   floor_node, sizes_node, interpolate});
//        replace_node(concat, interpolate);
//
//        return true;
//    };
//
//    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
//    register_matcher(m, callback);
//}

namespace {
using namespace ngraph;

bool compatible_attrs(const opset8::Interpolate::InterpolateAttrs& fst_attrs, const opset8::Interpolate::InterpolateAttrs& snd_attrs) {
    return std::tie(fst_attrs.mode, fst_attrs.shape_calculation_mode, fst_attrs.pads_begin, fst_attrs.pads_end,
                    fst_attrs.coordinate_transformation_mode, fst_attrs.nearest_mode, fst_attrs.antialias, fst_attrs.cube_coeff) ==
           std::tie(snd_attrs.mode, snd_attrs.shape_calculation_mode, snd_attrs.pads_begin, snd_attrs.pads_end,
                    snd_attrs.coordinate_transformation_mode, snd_attrs.nearest_mode, snd_attrs.antialias, snd_attrs.cube_coeff);
}

bool compatible_axes(const std::vector<int64_t>& fst_axes_vector, const std::vector<int64_t>& snd_axes_vector) {
    std::set<int64_t> fst_axes_set(fst_axes_vector.begin(), fst_axes_vector.end());
    for (const auto a : snd_axes_vector) {
        if (fst_axes_set.count(a) != 0) return false;
    }
    return true;
}

bool shape_calculation_mode_can_use_constant_inputs(const std::shared_ptr<opset8::Interpolate>& interpolate) {
    const auto& attrs = interpolate->get_attrs();
    if (attrs.shape_calculation_mode == ngraph::opset8::Interpolate::ShapeCalcMode::SIZES) {
        return std::dynamic_pointer_cast<opset8::Constant>(interpolate->input_value(1)) != nullptr;
    }
    return std::dynamic_pointer_cast<opset8::Constant>(interpolate->input_value(2)) != nullptr;
}

bool is_candidate_for_fusion(const std::shared_ptr<opset8::Interpolate>& interpolate) {
    return (interpolate->get_input_partial_shape(0).rank().is_static()) &&
           (interpolate->inputs().size() != 4 || std::dynamic_pointer_cast<opset8::Constant>(interpolate->input_value(3))) &&
           shape_calculation_mode_can_use_constant_inputs(interpolate);
}

std::vector<int64_t> get_interpolated_axes(const std::shared_ptr<opset8::Interpolate>& interpolate) {
    if (interpolate->inputs().size() != 4) {
        const auto input_rank = interpolate->get_input_partial_shape(0).rank().get_length();

        std::vector<int64_t> default_value(input_rank);
        std::iota(default_value.begin(), default_value.end(), 0);

        return default_value;
    }
    return std::dynamic_pointer_cast<opset8::Constant>(interpolate->input_value(3))->cast_vector<int64_t>();
}

bool can_be_fused(const std::shared_ptr<opset8::Interpolate>& fst, const std::shared_ptr<opset8::Interpolate>& snd) {
    if (!compatible_attrs(fst->get_attrs(), snd->get_attrs()) || !is_candidate_for_fusion(fst) || !is_candidate_for_fusion(snd) ||
        !compatible_axes(get_interpolated_axes(fst), get_interpolated_axes(snd)) {
        return false;
    }

    for (const auto& output : fst->outputs()) {
        for (const auto& consumer : output.get_target_inputs()) {
            if (consumer.get_node() != snd.get()) return false;
        }
    }

    return true;
}

ngraph::NodeVector subgraph_for_sizes_calculation_mode(const std::shared_ptr<opset8::Interpolate>& fst, const std::shared_ptr<opset8::Interpolate>& snd) {
    const auto fst_axes = get_interpolated_axes(fst);
    const auto snd_axes = get_interpolated_axes(snd);
    const auto fst_sizes = std::dynamic_pointer_cast<opset8::Constant>(fst->input_value(1))->cast_vector<int64_t>();
    const auto snd_sizes = std::dynamic_pointer_cast<opset8::Constant>(snd->input_value(1))->cast_vector<int64_t>();

    std::vector<std::pair<int64_t, int64_t>> axes_and_sizes;
    for (size_t i = 0; i < fst_axes.size(); ++i) {
        axes_and_sizes.emplace_back({fst_axes[i], fst_sizes[i]});
    }
    for (size_t i = 0; i < snd_axes.size(); ++i) {
        axes_and_sizes.emplace_back({snd_axes[i], snd_sizes[i]});
    }
    std::sort(axes_and_sizes.begin(),
              axes_and_sizes.end(),
              [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
                  return a.first < b.first;
              });
    std::vector<int64_t> new_axes;
    std::vector<int64_t> new_sizes;
    for (const auto& as : axes_and_sizes) {
        new_axes.emplace_back(as.first);
        new_sizes.emplace_back(as.second);
    }

    auto new_sizes_node = opset8::Constant::create(element::i64, {new_sizes.size()}, new_sizes);
    auto new_axes_node = opset8::Constant::create(element::i64, {new_axes.size()}, new_axes);
    auto new_sizes_cast = std::make_shared<opset8::Convert>(new_sizes_node, element::f32);
    auto shape_node = std::make_shared<opset8::ShapeOf>(fst->input_value(0));

    auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
    auto gather_node = std::make_shared<opset8::ShapeOf>(shape_node, new_axes_node, gather_axis_node);
    auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);

    auto div_node = std::make_shared<opset8::Divide>(new_sizes_cast, cast_shape_to_float);

    auto new_interpolate = fst->clone_with_new_inputs({fst->input_value(0), new_sizes_node, div_node, new_axes_node});

    return {new_sizes_node, new_axes_node, new_sizes_cast, shape_node, gather_axis_node, gather_node, cast_shape_to_float, div_node, new_interpolate};
}

ngraph::NodeVector subgraph_for_scales_calculation_mode(const std::shared_ptr<opset8::Interpolate>& fst, const std::shared_ptr<opset8::Interpolate>& snd) {
    const auto fst_axes = get_interpolated_axes(fst);
    const auto snd_axes = get_interpolated_axes(snd);
    const auto fst_scales = std::dynamic_pointer_cast<opset8::Constant>(fst->input_value(2))->cast_vector<float>();
    const auto snd_scales = std::dynamic_pointer_cast<opset8::Constant>(snd->input_value(2))->cast_vector<float>();

    std::vector<std::pair<int64_t, float>> axes_and_scales;
    for (size_t i = 0; i < fst_axes.size(); ++i) {
        axes_and_scales.emplace_back({fst_axes[i], fst_sizes[i]});
    }
    for (size_t i = 0; i < snd_axes.size(); ++i) {
        axes_and_scales.emplace_back({snd_axes[i], snd_sizes[i]});
    }
    std::sort(axes_and_scales.begin(),
              axes_and_scales.end(),
              [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {
                  return a.first < b.first;
              });
    std::vector<int64_t> new_axes;
    std::vector<float> new_scales;
    for (const auto& as : axes_and_sizes) {
        new_axes.emplace_back(as.first);
        new_scales.emplace_back(as.second);
    }

    auto new_scales_node = opset8::Constant::create(element::f32, {new_scales.size()}, new_scales);
    auto new_axes_node = opset8::Constant::create(element::i64, {new_axes.size()}, new_axes);
    auto shape_node = std::make_shared<opset8::ShapeOf>(fst->input_value(0));

    auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
    auto gather_node = std::make_shared<opset8::ShapeOf>(shape_node, new_axes_node, gather_axis_node);
    auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);

    auto mul_node = std::make_shared<opset8::Multiply>(cast_shape_to_float, new_scales_node);
    auto eps_node = opset8::Constant::create(element::f32, {}, std::vector<float>{1.0e-5f});
    auto add_node = std::make_shared<opset8::Multiply>(mul_node, eps_node);
    auto floor_node = std::make_shared<opset8::Floor>(add_node);
    auto cast_mul_result_to_int = std::make_shared<opset8::Convert>(floor_node, element::i64);

    auto new_interpolate = fst->clone_with_new_inputs({fst->input_value(0), cast_mul_result_to_int, new_scales_node, new_axes_node});

    return {new_sizes_node, new_axes_node, shape_node, gather_axis_node, gather_node, cast_shape_to_float, mul_node, eps_node,
            add_node, floor_node, cast_mul_result_to_int, new_interpolate};
}
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::InterpolateSequenceFusion, "InterpolateSequenceFusion", 0);

ngraph::pass::InterpolateSequenceFusion::InterpolateSequenceFusion() {
    MATCHER_SCOPE(InterpolateSequenceFusion);
    auto interpolate_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Interpolate>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto snd_interpolate = std::dynamic_pointer_cast<opset8::Interpolate>(m.get_match_root());
        if (!snd_interpolate) return false;

        auto fst_interpolate = std::dynamic_pointer_cast<opset8::Interpolate>(snd_interpolate->input_value(0));
        if (!fst_interpolate) return false;

        if (!can_be_fused(snd_interpolate, snd_interpolate)) return false;

        NodeVector new_subgraph;
        if (fst_interpolate->get_attrs().shape_calculation_mode == ngraph::opset8::Interpolate::ShapeCalcMode::SIZES) {
            new_subgraph = subgraph_for_sizes_calculation_mode(fst_interpolate, snd_interpolate);
        } else {
            new_subgraph = subgraph_for_scales_calculation_mode(fst_interpolate, snd_interpolate);
        }

        auto& new_interpolate = new_subgraph.back();
        new_interpolate->set_friendly_name(snd_interpolate->get_friendly_name());
        copy_runtime_info({fst_interpolate, snd_interpolate}, new_subgraph);
        replace_node(snd_interpolate, new_interpolate);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate_pattern, matcher_name);
    register_matcher(m, callback);
}
