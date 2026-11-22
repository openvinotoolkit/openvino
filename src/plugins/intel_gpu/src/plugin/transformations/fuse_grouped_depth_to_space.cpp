// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_grouped_depth_to_space.hpp"

#include <limits>
#include <optional>

#include "intel_gpu/op/grouped_depth_to_space.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

std::optional<std::vector<int64_t>> get_i64_values(const ov::Output<ov::Node>& output) {
    const auto constant = ov::util::get_constant_from_source(output);
    if (!constant) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>();
}

bool has_values(const ov::Output<ov::Node>& output, const std::vector<int64_t>& expected) {
    const auto values = get_i64_values(output);
    return values && *values == expected;
}

bool has_static_shape_values(const ov::Output<ov::Node>& output, const ov::PartialShape& expected_shape) {
    if (!expected_shape.is_static()) {
        return false;
    }

    const auto shape = expected_shape.to_shape();
    return has_values(output, std::vector<int64_t>(shape.begin(), shape.end()));
}

bool has_single_consumer(const std::shared_ptr<ov::Node>& node) {
    return node->output(0).get_target_inputs().size() == 1;
}

bool multiply_uses_factor(const ov::Output<ov::Node>& output, int64_t factor) {
    const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(output.get_node_shared_ptr());
    if (!multiply) {
        return false;
    }
    return has_values(multiply->input_value(0), {factor}) || has_values(multiply->input_value(1), {factor});
}

bool validate_repeat_indices(const std::shared_ptr<ov::op::v8::Gather>& gather, int64_t input_channels, int64_t repeats, ov::NodeVector& matched_nodes) {
    if (gather->get_batch_dims() != 0 || !has_values(gather->input_value(2), {1})) {
        return false;
    }

    if (const auto indices = get_i64_values(gather->input_value(1))) {
        if (indices->size() != static_cast<size_t>(input_channels * repeats)) {
            return false;
        }
        for (size_t index = 0; index < indices->size(); ++index) {
            if ((*indices)[index] != static_cast<int64_t>(index) / repeats) {
                return false;
            }
        }
        matched_nodes.push_back(gather->get_input_node_shared_ptr(1));
        return true;
    }

    const auto indices_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(gather->get_input_node_shared_ptr(1));
    if (!indices_reshape || indices_reshape->get_special_zero() || !has_single_consumer(indices_reshape) ||
        !has_values(indices_reshape->input_value(1), {-1})) {
        return false;
    }

    const auto indices_transpose = ov::as_type_ptr<ov::op::v1::Transpose>(indices_reshape->get_input_node_shared_ptr(0));
    if (!indices_transpose || !has_single_consumer(indices_transpose) || !has_values(indices_transpose->input_value(1), {1, 0})) {
        return false;
    }

    const auto tile = ov::as_type_ptr<ov::op::v0::Tile>(indices_transpose->get_input_node_shared_ptr(0));
    if (!tile || !has_single_consumer(tile) || !has_values(tile->input_value(1), {repeats, 1})) {
        return false;
    }

    const auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(tile->get_input_node_shared_ptr(0));
    if (!unsqueeze || !has_single_consumer(unsqueeze) || !has_values(unsqueeze->input_value(1), {0})) {
        return false;
    }

    const auto range = ov::as_type_ptr<ov::op::v4::Range>(unsqueeze->get_input_node_shared_ptr(0));
    if (!range || !has_single_consumer(range) || !has_values(range->input_value(0), {0}) || !has_values(range->input_value(1), {input_channels}) ||
        !has_values(range->input_value(2), {1})) {
        return false;
    }

    matched_nodes.insert(matched_nodes.end(), {range, unsqueeze, tile, indices_transpose, indices_reshape});
    return true;
}

bool validate_reshape_shapes(const std::shared_ptr<ov::op::v1::Reshape>& factor_reshape,
                             const std::shared_ptr<ov::op::v1::Reshape>& output_reshape,
                             int64_t input_time,
                             int64_t output_channels,
                             int64_t factor_t,
                             int64_t factor_s) {
    if (has_static_shape_values(factor_reshape->input_value(1), factor_reshape->get_output_partial_shape(0)) &&
        has_static_shape_values(output_reshape->input_value(1), output_reshape->get_output_partial_shape(0))) {
        return true;
    }

    const auto factor_shape = ov::as_type_ptr<ov::op::v0::Concat>(factor_reshape->get_input_node_shared_ptr(1));
    if (!factor_shape || factor_shape->get_axis() != 0 || factor_shape->get_input_size() != 7 || !has_values(factor_shape->input_value(1), {output_channels}) ||
        !has_values(factor_shape->input_value(2), {factor_t}) || !has_values(factor_shape->input_value(3), {factor_s}) ||
        !has_values(factor_shape->input_value(4), {factor_s}) || !has_values(factor_shape->input_value(5), {input_time})) {
        return false;
    }

    const auto output_shape = ov::as_type_ptr<ov::op::v0::Concat>(output_reshape->get_input_node_shared_ptr(1));
    return output_shape && output_shape->get_axis() == 0 && output_shape->get_input_size() == 5 &&
           has_values(output_shape->input_value(1), {output_channels}) && has_values(output_shape->input_value(2), {input_time * factor_t}) &&
           multiply_uses_factor(output_shape->input_value(3), factor_s) && multiply_uses_factor(output_shape->input_value(4), factor_s);
}

}  // namespace

FuseGroupedDepthToSpace::FuseGroupedDepthToSpace() {
    using namespace ov::pass::pattern;

    auto input_m = any_input(type_matches_any({ov::element::f16, ov::element::f32, ov::element::u8, ov::element::i8}));
    auto gather_m = wrap_type<ov::op::v8::Gather>({input_m, any_input(), wrap_type<ov::op::v0::Constant>()}, consumers_count(1));
    auto factor_reshape_m = wrap_type<ov::op::v1::Reshape>({gather_m, any_input()}, consumers_count(1));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({factor_reshape_m, wrap_type<ov::op::v0::Constant>()}, consumers_count(1));
    auto output_reshape_m = wrap_type<ov::op::v1::Reshape>({transpose_m, any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto input = pattern_map.at(input_m);
        const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_m).get_node_shared_ptr());
        const auto factor_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(factor_reshape_m).get_node_shared_ptr());
        const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_m).get_node_shared_ptr());
        const auto output_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(output_reshape_m).get_node_shared_ptr());

        if (transformation_callback(output_reshape) || output_reshape->get_users().size() != 1 ||
            !has_values(transpose->input_value(1), {0, 1, 5, 2, 6, 3, 7, 4})) {
            return false;
        }

        const auto input_shape = input.get_partial_shape();
        const auto factor_shape = factor_reshape->get_output_partial_shape(0);
        if (input_shape.rank() != 5 || factor_shape.rank() != 8 || !input_shape[1].is_static() || !input_shape[2].is_static() || !factor_shape[1].is_static() ||
            !factor_shape[2].is_static() || !factor_shape[3].is_static() || !factor_shape[4].is_static() || factor_shape[3] != factor_shape[4] ||
            factor_shape[5] != input_shape[2] || factor_shape[6] != input_shape[3] || factor_shape[7] != input_shape[4]) {
            return false;
        }

        const int64_t input_channels = input_shape[1].get_length();
        const int64_t input_time = input_shape[2].get_length();
        const int64_t output_channels = factor_shape[1].get_length();
        const int64_t factor_t = factor_shape[2].get_length();
        const int64_t factor_s = factor_shape[3].get_length();
        const int64_t factor = factor_t * factor_s * factor_s;
        if (input_channels <= 0 || output_channels <= 0 || factor_t <= 0 || factor_s <= 0 || (output_channels * factor) % input_channels != 0) {
            return false;
        }
        const int64_t repeats = output_channels * factor / input_channels;

        const auto gather_shape = gather->get_output_partial_shape(0);
        if (gather_shape.rank() != 5 || !gather_shape[1].is_static() || gather_shape[1].get_length() != input_channels * repeats ||
            !validate_reshape_shapes(factor_reshape, output_reshape, input_time, output_channels, factor_t, factor_s)) {
            return false;
        }

        ov::NodeVector matched_nodes = {gather, factor_reshape, transpose, output_reshape};
        if (!validate_repeat_indices(gather, input_channels, repeats, matched_nodes)) {
            return false;
        }

        size_t crop_begin_t = 0;
        std::shared_ptr<ov::Node> root = output_reshape;
        const auto output_user = output_reshape->get_users().front();
        if (const auto slice = ov::as_type_ptr<ov::op::v8::Slice>(output_user)) {
            if (!has_values(slice->input_value(1), {factor_t - 1}) || !has_values(slice->input_value(2), {std::numeric_limits<int64_t>::max()}) ||
                !has_values(slice->input_value(3), {1}) || !has_values(slice->input_value(4), {2})) {
                return false;
            }
            crop_begin_t = static_cast<size_t>(factor_t - 1);
            root = slice;
            matched_nodes.push_back(slice);
        }

        auto grouped = std::make_shared<ov::intel_gpu::op::GroupedDepthToSpace>(input,
                                                                                static_cast<size_t>(factor_t),
                                                                                static_cast<size_t>(factor_s),
                                                                                static_cast<size_t>(output_channels),
                                                                                crop_begin_t);
        if (!grouped->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
            return false;
        }

        grouped->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info(matched_nodes, grouped);
        ov::replace_node(root, grouped);
        return true;
    };

    auto m = std::make_shared<Matcher>(output_reshape_m, "FuseGroupedDepthToSpace");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
