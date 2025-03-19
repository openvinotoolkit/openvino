// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace {
// This function creates a partition of its argument into groups consisting of adjacent identical elements.
// Argument: std::vector<uint64_t> v
// Returns: partition of the argument
std::vector<std::vector<uint64_t>> grouped_vector(const std::vector<uint64_t>& v) {
    std::vector<std::vector<uint64_t>> result;

    if (v.empty())
        return result;

    uint64_t prev = v[0];
    std::vector<uint64_t> group;

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

std::pair<std::shared_ptr<ov::op::v1::Split>, uint64_t> get_split_before_concat(
    const std::shared_ptr<ov::op::v0::Concat>& concat) {
    // This function gets producers of the 'concat' node, checks that the following conditions are fulfilled:
    // 1) all producers for 'concat' are Split nodes;
    // 2) 'concat' has only one unique producer ('split');
    // 3) 'split' node has only one consumer;
    // 4) for any output port of 'split', number of corresponding input ports of the consumer is the same;
    // 5) for any output port 'i' of the 'split', corresponding input ports of the consumer are
    //    [i * m, ..., i * m + (m - 1)], where 'm' is the same for all 'i';
    // and, if all these conditions are fulfilled, returns the above mentioned 'Concat' node. Otherwise, if some of
    // these conditions is false, this functions returns nullptr.

    std::vector<uint64_t> idx;
    std::shared_ptr<ov::op::v1::Split> split;
    for (const auto& input : concat->input_values()) {
        // If 'concat' has some non-Split producer, then the transformation is not applicable.
        auto split_op = ov::as_type_ptr<ov::op::v1::Split>(input.get_node_shared_ptr());
        if (!split)
            split = split_op;
        if (!split_op || split != split_op)
            return {};
        idx.emplace_back(static_cast<uint64_t>(input.get_index()));
    }

    // If 'split' node has more than one consumer, then the transformation is not applicable.
    for (const auto& output : split->outputs()) {
        // if there is 'split' output port with no consumers,
        // SplitConcatPairToInterpolateFusion is not applicable
        if (output.get_target_inputs().empty()) {
            return {};
        }
        for (const auto& consumer : output.get_target_inputs()) {
            if (consumer.get_node() != concat.get())
                return {};
        }
    }

    // If numbers of consumer ports are various for various output ports of 'split', then the transformation is not
    // applicable.
    auto grouped_idx = grouped_vector(idx);
    std::unordered_set<uint64_t> sizes_of_groups;
    for (const auto& group : grouped_idx) {
        sizes_of_groups.insert(static_cast<uint64_t>(group.size()));
    }
    if (sizes_of_groups.size() != 1)
        return {};
    uint64_t size_of_group = *(sizes_of_groups.begin());

    // The transformation is applicable if output port 0 of 'split' goes to ports [0, ..., m-1] of next node,
    // output port 1 of 'split' goes to ports [m, ..., m + (m-1)] of next node, ..., output port i of 'split'
    // goes to ports [i * m, ..., i * m + (m - 1)], and so on.
    for (uint64_t i = 0; i < static_cast<uint64_t>(grouped_idx.size()); ++i) {
        const auto& current_group = grouped_idx[i];
        if (std::any_of(current_group.begin(), current_group.end(), [i](uint64_t j) {
                return j != i;
            })) {
            return {};
        }
    }

    return {split, size_of_group};
}
}  // namespace

ov::pass::SplitConcatPairToInterpolateFusion::SplitConcatPairToInterpolateFusion(bool use_shape_for_elimination) {
    MATCHER_SCOPE(SplitConcatPairToInterpolateFusion);
    // This transformation looks for Interpolate layer implemented using simple operations, namely Split and Concat,
    // and replaces found pattern with a sequence of Shape, StridedSlice, Const, Mul, Interpolate.
    // Found pattern:
    //     Split -> Concat
    // Here we assume that
    //     1) input data of Split is 4D or 5D tensor;
    //     2) split dimensions for 'split' belongs to {1, 2, 3};
    //     3) all outputs of 'split' go to only inputs of 'concat';
    //     4) 'concat' takes inputs only from 'split';
    //     5) split_dim of 'split' is equal to axis of 'concat';
    //     6) output port 0 of 'split' goes to ports [0, ..., m-1] of next node, output port 1 of 'split' goes to ports
    //        [m, ..., m + (m-1)] of next node, ..., output port i of 'split' goes to ports [i * m, ..., i * m + (m -
    //        1)], and so on;
    //     7) number of outputs of 'split' is equal to the length of the split axis.
    // Such subgraph
    //     Split -> Concat
    // can be replaced with the Interpolate layer with the following attributes:
    //     mode = nearest
    //     shape_calculation_mode = scales
    //     nearest_mode = round_prefer_floor
    //     pads_begin = {0}
    //     pads_end = {0}
    //     antialias = false
    //     coordinate_transformation_mode = half_pixel
    //     cube_coeff = -0.75
    // Next, the scaling factor in Interpolate is equal to a quotient of dividing number of input ports of 'concat'
    // by number of output ports of 'split'.
    //
    // Detect only concat, because we don't know how many inputs will go into concat.
    auto concat_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Concat>();
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        if (!concat)
            return false;

        uint64_t scale_factor;
        std::shared_ptr<ov::op::v1::Split> split;
        std::tie(split, scale_factor) = get_split_before_concat(concat);
        // If scale_factor == 1, then output data of Interpolate are equal to input data. Hence, we should not replace
        // Split->Concat pair with Interpolate.
        if (!split || !scale_factor || scale_factor == 1)
            return false;

        if (split->get_input_partial_shape(0).rank().is_dynamic())
            return false;
        int64_t split_input_rank = split->get_input_partial_shape(0).rank().get_length();
        // If this transformation is applied in the case of the the rank is less than 4, we have a performance
        // degradation. And, at this time, we have no models with Split->Concat pattern when this transformation is
        // applicable and input rank of Split is greater than 5.
        if (split_input_rank != 4 && split_input_rank != 5)
            return false;

        auto split_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis_const)
            return false;

        int64_t axis = split_axis_const->cast_vector<int64_t>()[0];

        if (split->get_input_partial_shape(0)[axis].is_static() &&
            split->get_input_partial_shape(0)[axis].get_length() != static_cast<int64_t>(split->outputs().size()))
            return false;

        ov::op::v4::Interpolate::InterpolateAttrs attrs;

        attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
        attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto scales_node =
            ov::op::v0::Constant::create(element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ov::op::v3::ShapeOf>(split->input_value(0));

        auto sslice_begin = ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node =
            std::make_shared<ov::op::v1::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ov::op::v0::Convert>(strided_slice_node, element::f32);
        auto mul_node = std::make_shared<ov::op::v1::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ov::op::v0::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ov::op::v0::Convert>(floor_node, element::i64);

        std::shared_ptr<Node> sizes_node;

        if (use_shape_for_elimination) {
            sizes_node = ov::util::get_constant_from_source(cast_mul_result_to_int);
        } else {
            disable_constant_folding(shape_node);
        }

        if (!sizes_node)
            sizes_node = cast_mul_result_to_int;

        auto interpolate = register_new_node<ov::op::v4::Interpolate>(split->input_value(0),
                                                                      sizes_node,
                                                                      scales_node,
                                                                      axis_node,
                                                                      attrs);

        interpolate->set_friendly_name(concat->get_friendly_name());
        copy_runtime_info({split, concat},
                          {scales_node,
                           axis_node,
                           shape_node,
                           sslice_begin,
                           sslice_end,
                           strided_slice_node,
                           cast_shape_to_float,
                           mul_node,
                           floor_node,
                           sizes_node,
                           interpolate});
        replace_node(concat, interpolate);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}
