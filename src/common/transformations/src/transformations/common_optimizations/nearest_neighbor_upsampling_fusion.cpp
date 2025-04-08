// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
using namespace ov;

// In the transformation, a constant for Multiply must have the following shape:
//      [1, 1, S_1, 1, S_2, ..., 1, S_i, ..., 1, S_{r - 2}, 1], (1)
// where (r - 2) is number of spatial axes, and each S_i is a scale for the axis i.
// This function returns the vector with elements
//      [S_1, S_2, ..., S_i, ..., S_{r - 2}],
// when the shape, 's', has the form (1), and the empty vector otherwise.
std::vector<float> get_scales_from_mul_const_shape(const Shape& s, uint64_t input_rank) {
    if (input_rank < 4 || s.size() != 2 * input_rank - 2)
        return {};

    ov::Shape expected_shape(2 * input_rank - 2, 1);
    std::vector<float> scales(input_rank - 2);
    for (uint64_t i = 1; i <= input_rank - 2; ++i) {
        expected_shape[2 * i] = s[2 * i];
        scales[i - 1] = static_cast<float>(s[2 * i]);
    }

    if (s != expected_shape)
        return {};

    return scales;
}

bool check_concat_1(const std::shared_ptr<ov::op::v0::Concat>& concat, const Shape& shape) {
    size_t rank = shape.size();

    const auto inputs = concat->input_values();
    size_t num_of_input_values = inputs.size();

    if (num_of_input_values != 2 * rank - 2)
        return false;

    std::vector<int64_t> input_constants(num_of_input_values, 1);
    for (size_t i = 1; i < num_of_input_values; ++i) {
        const auto& current_input = ov::as_type_ptr<ov::op::v0::Unsqueeze>(inputs[i].get_node_shared_ptr());
        if (!current_input)
            return false;

        const auto current_input_axis =
            ov::as_type_ptr<ov::op::v0::Constant>(current_input->input_value(1).get_node_shared_ptr());
        if (!current_input_axis || current_input_axis->cast_vector<int64_t>() != std::vector<int64_t>{0})
            return false;

        const auto unsqueezed_const =
            ov::as_type_ptr<ov::op::v0::Constant>(current_input->input_value(0).get_node_shared_ptr());
        if (!unsqueezed_const)
            return false;

        const auto unsqueezed_const_value = unsqueezed_const->cast_vector<int64_t>();
        if (unsqueezed_const_value.size() != 1)
            return false;

        input_constants[i] = unsqueezed_const_value[0];
    }

    std::vector<int64_t> expected_input_constants(num_of_input_values, 1);
    for (size_t i = 1; i <= rank - 2; ++i) {
        expected_input_constants[2 * i - 1] = static_cast<int64_t>(shape[i]);
    }
    expected_input_constants.back() = static_cast<int64_t>(shape.back());

    if (input_constants != expected_input_constants)
        return false;

    return true;
}

// In this transformation 'concat_2' must have r inputs (where r is an output rank of the root of the transformation
// pattern). And (r - 1) inputs must be unsqueezed constants, and the list of these constants is
//      [newD_1, newD_2, ..., newD_{r - 2}, C],
// where C is number of channels in the output shape of the root of the transformation pattern.
//
// This function gets a new spatial shape from unsqueezed constants of 'concat_2', that is, the vector with elements
//      [newD_1, newD_2, ..., newD_{r - 2}].
std::vector<int64_t> get_new_spatial_shape_from_concat_2(const std::shared_ptr<ov::op::v0::Concat>& concat,
                                                         const Shape& input_shape) {
    size_t rank = input_shape.size();

    const auto inputs = concat->input_values();
    size_t num_of_input_values = inputs.size();

    if (num_of_input_values != rank)
        return {};

    std::vector<int64_t> input_constants(num_of_input_values - 1, 0);

    for (size_t i = 1; i < num_of_input_values; ++i) {
        const auto& current_input = ov::as_type_ptr<ov::op::v0::Unsqueeze>(inputs[i].get_node_shared_ptr());
        if (!current_input)
            return {};

        const auto current_input_axis =
            ov::as_type_ptr<ov::op::v0::Constant>(current_input->input_value(1).get_node_shared_ptr());
        if (!current_input_axis || current_input_axis->cast_vector<int64_t>() != std::vector<int64_t>{0})
            return {};

        const auto unsqueezed_const =
            ov::as_type_ptr<ov::op::v0::Constant>(current_input->input_value(0).get_node_shared_ptr());
        if (!unsqueezed_const)
            return {};

        const auto unsqueezed_const_value = unsqueezed_const->cast_vector<int64_t>();
        if (unsqueezed_const_value.size() != 1)
            return {};

        input_constants[i - 1] = unsqueezed_const_value[0];
    }

    if (input_constants.back() != static_cast<int64_t>(input_shape.back()))
        return {};

    input_constants.pop_back();

    return input_constants;
}
}  // namespace

ov::pass::NearestNeighborUpsamplingFusion::NearestNeighborUpsamplingFusion() {
    MATCHER_SCOPE(NearestNeighborUpsamplingFusion);
    // This transformation looks for Interpolate layer implemented using simple operations, namely ShapeOf,
    // StridedSlice, Concat, Reshape, Mul, and replaces found pattern with a sequence of Shape, StridedSlice, Const,
    // Mul, Interpolate. Found pattern (for 4D case, in a general case the pattern is similar):
    //
    //  |---------|
    //  |   op    |
    //  |---|-----|
    //      |  shape: [N, H, W, C]                |-----------|        |----------------|
    //      |------------------------------------>|0 ShapeOf  |------->|0 StridedSlice  |
    //      |                                     |-----------|        |-------|--------|
    //      |                                                                  |
    //      |                                                                  |
    //      |                                               |------------------|---------------------------|
    //      |                                               |                                              |
    //      |                                               |                                              |
    //      |                                               |                                              |
    //      |                                               |                                              |
    //      |      |---------------|                        |                  |---------------|           |
    //      |      | Concat        |                        |                  | Concat        |           |
    //      |      | (concat_1)    |                        |                  | (concat_2)    |           |
    //      |      |               |                        |                  |               |           |
    //      |      |              0|<-----------------------|                  |              0|<----------|
    //      |      |               |                                           |               |
    //      |      |               |                                           |               |
    //      |      |               |      |-------------|   |------------|     |               |      |-------------|
    //      |--------------| |      |               |      |             |   | Constant   |     |               |      |
    //      |   | Constant     | |      |               |      |            0|<--| value: H   |     |               | |
    //      0|<--| value: new_H | |      |               |      |             |   |------------|     |               |
    //      |             |   |--------------| |      |               |      |             |                      | | |
    //      | |      |               |      | Unsqueeze   |   |------------|     |               |      | Unsqueeze   |
    //      |------------| |      |               |      |             |   | Constant   |     |               |      |
    //      |   | Constant   | |      |              1|<-----|            1|<--| value: 0   |     | 1|<-----| 1|<--|
    //      value: 0   | |      |               |      |-------------|   |------------|     |               |
    //      |-------------|   |------------| |      |               |                                           | | | |
    //      |      |-------------|   |------------|     |               |      |-------------|   |--------------| | | |
    //      |             |   | Constant   |     |               |      |             |   | Constant     | |      | | |
    //      0|<--| value: 1   |     |               |      |            0|<--| value: new_W | |      |               |
    //      |             |   |------------|     |               |      |             |   |--------------| |      | | |
    //      |                      |               |      |             | |      |               |      | Unsqueeze   |
    //      |------------|     |               |      | Unsqueeze   |   |------------| |      |               |      |
    //      |   | Constant   |     |               |      |             |   | Constant   | |      | 2|<-----| 1|<--|
    //      value: 0   |     |              2|<-----|            1|<--| value: 0   | |      |               |
    //      |-------------|   |------------|     |               |      |-------------|   |------------| |      | | | |
    //      |      |               |      |-------------|   |------------|     |               |      |-------------|
    //      |------------| |      |               |      |             |   | Constant   |     |               |      |
    //      |   | Constant   | |      |               |      |            0|<--| value: W   |     |               | |
    //      0|<--| value: C   | |      |               |      |             |   |------------|     |               | |
    //      |   |------------| |      |               |      |             |                      |               | | |
    //      |      |               |      | Unsqueeze   |   |------------|     |               |      | Unsqueeze   |
    //      |------------| |      |               |      |             |   | Constant   |     |               |      |
    //      |   | Constant   | |      |              3|<-----|            1|<--| value: 0   |     | 3|<-----| 1|<--|
    //      value: 0   | |      |               |      |-------------|   |------------|     |------|--------|
    //      |-------------|   |------------| |      |               |                                                  |
    //      |      |               |      |-------------|   |------------|            |
    //      |      |               |      |             |   | Constant   |            |
    //      |      |               |      |            0|<--| value: 1   |            |
    //      |      |               |      |             |   |------------|            |
    //      |      |               |      |             |                             |
    //      |      |               |      | Unsqueeze   |   |------------|            |
    //      |      |               |      |             |   | Constant   |            |
    //      |      |              4|<-----|            1|<--| value: 0   |            |
    //      |      |               |      |-------------|   |------------|            |
    //      |      |               |                                                  |
    //      |      |               |      |-------------|   |------------|            |
    //      |      |               |      |             |   | Constant   |            |
    //      |      |               |      |            0|<--| value: C   |            |
    //      |      |               |      |             |   |------------|            |
    //      |      |               |      |             |                             |
    //      |      |               |      | Unsqueeze   |   |------------|            |
    //      |      |               |      |             |   | Constant   |            |
    //      |      |              5|<-----|            1|<--| value: 0   |            |
    //      |      |------|--------|      |-------------|   |------------|            |
    //      |             |                                                           |
    //      |             |                                                           |
    //      |             |------------|                                              |
    //      |                          |                                              |
    //      |     |---------------|    |                                              |
    //      |     |  Reshape      |    |                                              |
    //      |---->|0 (reshape_1) 1|<---|                                              |
    //            |               |                                                   |
    //            |-----|---------|                                                   |
    //                  |                                                             |
    //       |----------|                                                             |
    //       |    |-------------|     |----------------|                              |
    //       |--->|0   Mul     1|<----|   Const        |                              |
    //            |   (mul)     |     |  (mul_const)   |                              |--------|
    //            |-------------|     |----------------|                                       |
    //                 |                                                                       |
    //                 |                                                                       |
    //                 |                                         |---------------------|       |
    //                 |                                         |    Reshape          |       |
    //                 |---------------------------------------->|0   (reshape_2)     1|<------|
    //                                                           |                     |
    //                                                           |---------------------|
    //
    // Here digits 0, 1, ..., are numbers of input ports of nodes.
    //
    // That is, we found the subgraph of the above mentioned form, where
    //      1) an output rank r of 'op' is greater or equal to 4;
    //      2) an output shape of 'op' has the form [N, D_1, D_2, ..., D_{r - 2}, C];
    //      3) unsqueezed constants for 'concat_1' are
    //          D_1 for the input port 1 of 'concat_1' and 1 for the input port 2 of 'concat_1';
    //          D_2 for the input port 3 of 'concat_1' and 1 for the input port 4 of 'concat_1';
    //          ...
    //          D_i for the input port 2 * (i - 1) + 1 of 'concat_1' and 1 for the input port 2 * i of 'concat_1';
    //          ...
    //          D_{r - 2} for the input port 2 * ((r - 2) - 1) + 1 of 'concat_1' and 1 for the input port 2 * (r - 2) of
    //          'concat_1'; C for the input port 2 * (r - 2) + 1 of 'concat_1';
    //      4) unsqueezed constants for 'concat_2' are
    //          newD_1 for the input port 1 of 'concat_1';
    //          newD_2 for the input port 2 of 'concat_1';
    //          ...
    //          newD_i for the input port i of 'concat_1';
    //          ...
    //          newD_{r - 2} for the input port (r - 2) of 'concat_1';
    //          C for the input port (r - 2) + 1 of 'concat_1';
    //      5) the shape of 'mul_const' is [1, 1, S_1, 1, S_2, ..., 1, S_i, ..., 1, S_{r - 2}, 1] where S_i is a scale
    //      for the axis i; 6) all elements of 'mul_const' are equal to 1.0.
    //
    // Such subgraph can be replaced by the Interpolate node with
    //      1) mode='nearest' and shape_calculation_mode='scales';
    //      2) 'sizes' input as a constant with the value [newD_1, newD_2, ..., newD_i, ..., newD_{r - 2}];
    //      3) 'scales' input as a constant with the value [S_1, S_2, ..., S_i, ..., S_{r - 2}];
    //      4) 'axes' input as a constant with the value [1, 2, ..., r - 2].
    //
    // Of course, the replacement shouldn't be done, if all S_i are equal to 1.
    auto input = pass::pattern::any_input(pattern::has_static_shape());
    auto concat_1 = pattern::wrap_type<ov::op::v0::Concat>();
    auto concat_2 = pattern::wrap_type<ov::op::v0::Concat>();
    auto reshape_1 = pattern::wrap_type<ov::op::v1::Reshape>({input, concat_1});
    auto mul_const = pattern::wrap_type<ov::op::v0::Constant>(pattern::has_static_shape());
    auto mul = pattern::wrap_type<ov::op::v1::Multiply>({reshape_1, mul_const});
    auto reshape_2 = pattern::wrap_type<ov::op::v1::Reshape>({mul, concat_2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto reshape_2_node =
            ov::as_type_ptr<ov::op::v1::Reshape>(pattern_to_output.at(reshape_2).get_node_shared_ptr());
        const auto mul_node = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr());
        if (!reshape_2_node || !mul_node)
            return false;

        const auto mul_const_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_const).get_node_shared_ptr());
        if (!mul_const_node)
            return false;

        const auto reshape_1_node =
            ov::as_type_ptr<ov::op::v1::Reshape>(pattern_to_output.at(reshape_1).get_node_shared_ptr());
        if (!reshape_1_node)
            return false;

        uint64_t input_rank = static_cast<uint64_t>(reshape_1_node->get_input_partial_shape(0).rank().get_length());
        const auto mul_const_shape = mul_const_node->get_output_shape(0);
        const auto scales = get_scales_from_mul_const_shape(mul_const_shape, input_rank);
        if (scales.empty() || std::all_of(scales.begin(), scales.end(), [](float s) {
                return s == 1.0f;
            })) {
            return false;
        }

        const auto mul_const_value = mul_const_node->cast_vector<float>();
        if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [](float x) {
                return x != 1.0f;
            })) {
            return false;
        }

        const auto concat_1_node =
            ov::as_type_ptr<ov::op::v0::Concat>(pattern_to_output.at(concat_1).get_node_shared_ptr());
        if (!concat_1_node)
            return false;

        const auto input_shape = reshape_1_node->get_input_shape(0);
        if (!check_concat_1(concat_1_node, input_shape))
            return false;

        const auto concat_2_node =
            ov::as_type_ptr<ov::op::v0::Concat>(pattern_to_output.at(concat_2).get_node_shared_ptr());
        if (!concat_2_node)
            return false;

        const auto new_spatial_shape = get_new_spatial_shape_from_concat_2(concat_2_node, input_shape);
        if (new_spatial_shape.empty())
            return false;

        const auto ss_before_concat_1 =
            ov::as_type_ptr<ov::op::v1::StridedSlice>(concat_1_node->input_value(0).get_node_shared_ptr());
        const auto ss_before_concat_2 =
            ov::as_type_ptr<ov::op::v1::StridedSlice>(concat_2_node->input_value(0).get_node_shared_ptr());
        if (!ss_before_concat_1 || !ss_before_concat_2 || ss_before_concat_1.get() != ss_before_concat_2.get())
            return false;

        const auto shapeof_node =
            ov::as_type_ptr<ov::op::v3::ShapeOf>(ss_before_concat_1->input_value(0).get_node_shared_ptr());
        if (!shapeof_node)
            return false;

        const auto before_shapeof = shapeof_node->input_value(0);
        const auto before_reshape_1 = reshape_1_node->input_value(0);
        if (before_shapeof.get_node() != before_reshape_1.get_node())
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

        const auto& input_node = pattern_to_output.at(input);
        const auto& type = input_node.get_element_type();
        const auto scales_node = ov::op::v0::Constant::create(type, {scales.size()}, scales);
        const auto sizes_node =
            ov::op::v0::Constant::create(element::i64, {new_spatial_shape.size()}, new_spatial_shape);

        std::vector<int64_t> axes(input_rank - 2);
        std::iota(axes.begin(), axes.end(), static_cast<int64_t>(1));
        const auto axes_node = ov::op::v0::Constant::create(element::i64, {axes.size()}, axes);

        auto interpolate =
            register_new_node<ov::op::v4::Interpolate>(before_shapeof, sizes_node, scales_node, axes_node, attrs);

        interpolate->set_friendly_name(reshape_2_node->get_friendly_name());
        copy_runtime_info(
            {reshape_2_node, mul_node, mul_const_node, concat_1_node, concat_2_node, ss_before_concat_1, shapeof_node},
            {scales_node, sizes_node, axes_node, interpolate});
        replace_node(reshape_2_node, interpolate);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_2, matcher_name);
    register_matcher(m, callback);
}
