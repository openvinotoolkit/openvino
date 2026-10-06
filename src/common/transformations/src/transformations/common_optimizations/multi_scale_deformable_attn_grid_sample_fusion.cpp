// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/multi_scale_deformable_attn_grid_sample_fusion.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/msda.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {
namespace {

using opset12::Add;
using opset12::Concat;
using opset12::Gather;
using opset12::GridSample;
using opset12::Multiply;
using opset12::ReduceSum;
using opset12::Reshape;
using opset12::Transpose;
using opset12::VariadicSplit;

template <typename T>
std::shared_ptr<T> input(const std::shared_ptr<ov::Node>& node, size_t index = 0) {
    return node && node->get_input_size() > index ? ov::as_type_ptr<T>(node->get_input_node_shared_ptr(index))
                                                  : nullptr;
}

// Frontend constant compression can insert Convert between a scalar and its user.
bool scalar_is(const std::shared_ptr<ov::Node>& node, float expected) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
        if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node))
            constant = input<ov::op::v0::Constant>(convert);
    }
    if (!constant || ov::shape_size(constant->get_shape()) != 1)
        return false;
    return constant->cast_vector<float>()[0] == expected;
}

// Constant input `index` of `node` holds exactly `expected`.
bool input_values_are(const std::shared_ptr<ov::Node>& node, size_t index, const std::vector<int64_t>& expected) {
    const auto constant = input<ov::op::v0::Constant>(node, index);
    return constant && constant->cast_vector<int64_t>() == expected;
}

bool has_shape(const std::shared_ptr<ov::Node>& node, const ov::Shape& expected) {
    const auto& ps = node->get_output_partial_shape(0);
    return ps.is_static() && ps.to_shape() == expected;
}

// Per level nodes whose layouts are checked once the MSDA dimensions are known.
struct LevelNodes {
    std::shared_ptr<ov::Node> image_flat, image, gather, squeeze, coords, level_shape;
    size_t h, w;
};

// The sampling-locations tensor has the full [B, Q, H, L, P, 2] layout.
bool has_msda_locations_layout(const std::shared_ptr<ov::Node>& node) {
    if (!node)
        return false;
    const auto& ps = node->get_output_partial_shape(0);
    return ps.is_static() && ps.size() == 6 && ps[5].get_length() == 2;
}

}  // namespace

MultiScaleDeformableAttnGridSampleFusion::MultiScaleDeformableAttnGridSampleFusion() {
    // The match root is Reshape(ReduceSum(Multiply(...))); the output projection
    // Transpose([0,2,1]) that follows it is checked in the callback.
    auto root = pattern::wrap_type<Reshape>({pattern::wrap_type<ReduceSum>(), pattern::any_input()});
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto reshape = ov::as_type_ptr<Reshape>(m.get_match_root());
        const auto reduce = input<ReduceSum>(reshape);
        const auto mul = input<Multiply>(reduce);
        const auto values_reshape = input<Reshape>(mul);
        const auto concat = input<Concat>(values_reshape);
        const auto weights_reshape = input<Reshape>(mul, 1);
        const auto weights_transpose = input<Transpose>(weights_reshape);
        // The weights operand only needs the [B,Q,H,L,P] layout; earlier
        // cleanup passes may have removed an identity Reshape on its path.
        const auto weights = weights_transpose ? weights_transpose->get_input_node_shared_ptr(0) : nullptr;
        if (!reduce || !mul || !concat || !values_reshape || !weights || !weights_reshape || !weights_transpose ||
            concat->get_axis() != -2 || concat->get_input_size() < 1 || reduce->get_keep_dims() ||
            !scalar_is(reduce->get_input_node_shared_ptr(1), -1) ||
            !input_values_are(weights_transpose, 1, {0, 2, 1, 3, 4}))
            return false;
        const size_t num_levels = concat->get_input_size();

        // The GPU pipeline lowers output_proj/MatMul(transpose_a=true) to an
        // explicit Transpose([0,2,1]) after Reshape. MSDA already emits [B,Q,H*D].
        const auto& users = reshape->output(0).get_target_inputs();
        if (users.size() != 1 || users.begin()->get_index() != 0)
            return false;
        const auto output_transpose = ov::as_type_ptr<Transpose>(users.begin()->get_node()->shared_from_this());
        const auto output_order = output_transpose ? input<ov::op::v0::Constant>(output_transpose, 1) : nullptr;
        if (!output_order || output_order->cast_vector<int64_t>() != std::vector<int64_t>({0, 2, 1}))
            return false;

        std::shared_ptr<ov::Node> value, locations;
        std::shared_ptr<VariadicSplit> split;
        std::vector<int32_t> spatial_shapes;
        std::vector<int32_t> level_starts;
        std::vector<LevelNodes> level_nodes;
        int32_t position = 0;
        for (size_t i = 0; i < num_levels; ++i) {
            // VariadicSplit(value[B,S,H,D]) -> Reshape -> Transpose -> Reshape
            // -> GridSample -> Unsqueeze (or Reshape), for each feature level.
            const auto level_shape = concat->get_input_node_shared_ptr(i);
            if (!ov::is_type<opset12::Unsqueeze>(level_shape) && !ov::is_type<Reshape>(level_shape))
                return false;
            const auto grid = input<GridSample>(level_shape);
            const auto image = input<Reshape>(grid);
            const auto image_transpose = input<Transpose>(image);
            const auto image_flat = input<Reshape>(image_transpose);
            const auto current_split = input<VariadicSplit>(image_flat);
            const auto coords = input<Reshape>(grid, 1);
            const auto coords_transpose = input<Transpose>(coords);
            // A Gather with a [1] shaped index keeps the level axis, which a
            // Squeeze (or an equivalent Reshape) then removes.
            std::shared_ptr<ov::Node> squeeze;
            auto gather = input<Gather>(coords_transpose);
            if (!gather && coords_transpose) {
                const auto node = coords_transpose->get_input_node_shared_ptr(0);
                if (ov::is_type<opset12::Squeeze>(node) || ov::is_type<Reshape>(node)) {
                    squeeze = node;
                    gather = input<Gather>(squeeze);
                }
            }
            const auto sub = input<Add>(gather);
            const auto twice = input<Multiply>(sub);
            const auto raw_locations = twice ? twice->get_input_node_shared_ptr(0) : nullptr;
            // GridSample consumes coordinates normalized to [-1,1], whereas MSDA
            // consumes [0,1] locations. Exports compute the coordinates as
            // Add(Multiply(x, 2), -1); when x has the full [B,Q,H,L,P,2]
            // layout, x itself is the [0,1] tensor, so the fusion rewires it
            // directly and the normalization is dropped.
            const bool direct = sub && twice && scalar_is(sub->get_input_node_shared_ptr(1), -1) &&
                                scalar_is(twice->get_input_node_shared_ptr(1), 2) &&
                                has_msda_locations_layout(raw_locations);
            const auto current_locations = direct ? raw_locations : nullptr;
            if (!grid || !image || !image_transpose || !image_flat || !current_split || !coords || !coords_transpose ||
                !gather || !sub || !twice || !current_locations)
                return false;
            const auto& attr = grid->get_attributes();
            if (attr.align_corners || attr.mode != GridSample::InterpolationMode::BILINEAR ||
                attr.padding_mode != GridSample::PaddingMode::ZEROS)
                return false;
            if (image_flat->input_value(0).get_index() != i || !grid->get_input_partial_shape(0).is_static())
                return false;
            // Value keys go to [B*H, D, h, w] through Transpose([0,2,1]), and
            // level i takes locations[:, :, :, i] through Transpose([0,2,1,3,4]).
            const auto axis = input<ov::op::v0::Constant>(gather, 2);
            const auto axis_value =
                axis && ov::shape_size(axis->get_shape()) == 1 ? axis->cast_vector<int64_t>()[0] : -1;
            if (!input_values_are(image_transpose, 1, {0, 2, 1}) ||
                !input_values_are(coords_transpose, 1, {0, 2, 1, 3, 4}) || (axis_value != 3 && axis_value != -3) ||
                gather->get_batch_dims() != 0 || !input_values_are(gather, 1, {static_cast<int64_t>(i)}))
                return false;
            // The level spatial shape is read from the GridSample image input.
            const auto image_shape = grid->get_input_shape(0);
            const auto h = static_cast<int32_t>(image_shape[2]);
            const auto w = static_cast<int32_t>(image_shape[3]);
            const auto& split_ps = current_split->get_output_partial_shape(i);
            if (h <= 0 || w <= 0 || !split_ps.is_static() || split_ps.size() < 2 ||
                split_ps[1].get_length() != static_cast<int64_t>(h) * w)
                return false;
            if (i == 0) {
                split = current_split;
                value = current_split->get_input_node_shared_ptr(0);
                locations = current_locations;
            } else if (split != current_split || locations != current_locations) {
                // All levels must sample the same value and locations tensors.
                return false;
            }
            spatial_shapes.insert(spatial_shapes.end(), {h, w});
            level_nodes.push_back({image_flat,
                                   image,
                                   gather,
                                   squeeze,
                                   coords,
                                   level_shape,
                                   static_cast<size_t>(h),
                                   static_cast<size_t>(w)});
            level_starts.push_back(position);
            position += h * w;
        }
        if (!value || !value->get_output_partial_shape(0).is_static() || value->get_output_shape(0).size() != 4 ||
            value->get_output_shape(0)[1] != static_cast<size_t>(position) ||
            !locations->get_output_partial_shape(0).is_static() || !weights->get_output_partial_shape(0).is_static())
            return false;
        // Heads and channels come from the value projection, levels and points
        // from the sampling pattern; no dimension is architecture specific.
        const auto value_shape = value->get_output_shape(0);
        const auto loc_shape = locations->get_output_shape(0);
        const auto weight_shape = weights->get_output_shape(0);
        if (loc_shape.size() != 6 || weight_shape.size() != 5 || loc_shape[0] != value_shape[0] ||
            loc_shape[2] != value_shape[2] || loc_shape[3] != num_levels || loc_shape[5] != 2 ||
            weight_shape != ov::Shape({loc_shape[0], loc_shape[1], loc_shape[2], num_levels, loc_shape[4]}) ||
            reshape->get_output_partial_shape(0) !=
                ov::PartialShape(ov::Shape{value_shape[0], value_shape[2] * value_shape[3], loc_shape[1]}))
            return false;
        // Each Reshape keeps the element order of the reference formulation,
        // which its output shape fully determines.
        const size_t batch = value_shape[0], heads = value_shape[2], embed = value_shape[3];
        const size_t queries = loc_shape[1], points = loc_shape[4];
        for (const auto& level : level_nodes) {
            // locations[:, :, :, i] is [B,Q,H,P,2]; a [1] index keeps the level axis until the squeeze.
            const bool gathered = level.squeeze ? has_shape(level.gather, {batch, queries, heads, 1, points, 2}) &&
                                                      has_shape(level.squeeze, {batch, queries, heads, points, 2})
                                                : has_shape(level.gather, {batch, queries, heads, points, 2});
            if (!gathered || !has_shape(level.image_flat, {batch, level.h * level.w, heads * embed}) ||
                !has_shape(level.image, {batch * heads, embed, level.h, level.w}) ||
                !has_shape(level.coords, {batch * heads, queries, points, 2}) ||
                !has_shape(level.level_shape, {batch * heads, embed, queries, 1, points}))
                return false;
        }
        if (!has_shape(values_reshape, {batch * heads, embed, queries, num_levels * points}) ||
            !has_shape(weights_reshape, {batch * heads, 1, queries, num_levels * points}))
            return false;

        auto shapes = opset12::Constant::create(ov::element::i32, ov::Shape{num_levels, 2}, spatial_shapes);
        auto starts = opset12::Constant::create(ov::element::i32, ov::Shape{num_levels}, level_starts);
        auto msda =
            std::make_shared<ov::op::internal::MSDA>(ov::OutputVector{value, shapes, starts, locations, weights});
        msda->set_friendly_name(output_transpose->get_friendly_name());
        ov::copy_runtime_info({reshape, output_transpose}, msda);
        ov::replace_node(output_transpose, msda);
        return true;
    };
    register_matcher(std::make_shared<pattern::Matcher>(root, "MultiScaleDeformableAttnGridSampleFusion"), callback);
}

}  // namespace ov::pass
