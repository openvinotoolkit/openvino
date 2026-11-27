// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_avg_down.hpp"

#include <algorithm>
#include <optional>

#include "intel_gpu/op/grouped_space_to_depth.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

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

bool is_zero_fill(const ov::Output<ov::Node>& output) {
    const auto constant = ov::util::get_constant_from_source(output);
    if (!constant || ov::shape_size(constant->get_shape()) != 1) {
        return false;
    }
    return constant->cast_vector<double>()[0] == 0.0;
}

bool has_vector_size(const ov::Output<ov::Node>& output, size_t expected) {
    const auto shape = output.get_partial_shape();
    return shape.is_static() && shape.size() == 1 && shape[0].get_length() == static_cast<int64_t>(expected);
}

bool same_source(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    return lhs.get_node() == rhs.get_node() && lhs.get_index() == rhs.get_index();
}

bool is_time_length(const ov::Output<ov::Node>& output, const ov::Output<ov::Node>& data) {
    const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(output.get_node_shared_ptr());
    if (!gather || gather->get_batch_dims() != 0 || !has_values(gather->input_value(1), {2}) || !has_values(gather->input_value(2), {0})) {
        return false;
    }

    const auto shape_of = ov::as_type_ptr<ov::op::v3::ShapeOf>(gather->get_input_node_shared_ptr(0));
    return shape_of && same_source(shape_of->input_value(0), data);
}

bool is_padding_formula(const std::shared_ptr<ov::Node>& node, const ov::Output<ov::Node>& data, int64_t factor_t) {
    const auto outer_mod = ov::as_type_ptr<ov::op::v1::FloorMod>(node);
    if (!outer_mod || !has_values(outer_mod->input_value(1), {factor_t})) {
        return false;
    }

    const auto is_time_remainder = [&](const ov::Output<ov::Node>& output) {
        const auto inner_mod = ov::as_type_ptr<ov::op::v1::FloorMod>(output.get_node_shared_ptr());
        return inner_mod && has_values(inner_mod->input_value(1), {factor_t}) && is_time_length(inner_mod->input_value(0), data);
    };

    const auto difference = outer_mod->input_value(0);
    if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(difference.get_node_shared_ptr())) {
        return has_values(subtract->input_value(0), {factor_t}) && is_time_remainder(subtract->input_value(1));
    }

    const auto add = ov::as_type_ptr<ov::op::v1::Add>(difference.get_node_shared_ptr());
    if (!add) {
        return false;
    }
    for (size_t factor_index = 0; factor_index < 2; ++factor_index) {
        if (!has_values(add->input_value(factor_index), {factor_t})) {
            continue;
        }
        const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(add->get_input_node_shared_ptr(1 - factor_index));
        if (!multiply) {
            continue;
        }
        for (size_t negative_index = 0; negative_index < 2; ++negative_index) {
            if (has_values(multiply->input_value(negative_index), {-1}) && is_time_remainder(multiply->input_value(1 - negative_index))) {
                return true;
            }
        }
    }
    return false;
}

bool is_padding_formula_value(const ov::Output<ov::Node>& output, const ov::Output<ov::Node>& data, int64_t factor_t) {
    if (is_padding_formula(output.get_node_shared_ptr(), data, factor_t)) {
        return true;
    }
    if (!has_vector_size(output, 1)) {
        return false;
    }
    const auto node = output.get_node_shared_ptr();
    if (ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v0::Squeeze>(node) || ov::is_type<ov::op::v1::Reshape>(node)) {
        return is_padding_formula_value(node->input_value(0), data, factor_t);
    }
    return false;
}

bool validate_padding_values(const std::shared_ptr<ov::op::v0::Concat>& padding_values, const ov::Output<ov::Node>& data, int64_t factor_t) {
    if (!padding_values || padding_values->get_axis() != 0) {
        return false;
    }

    size_t offset = 0;
    bool found_temporal_padding = false;
    for (const auto& input : padding_values->input_values()) {
        const auto shape = input.get_partial_shape();
        if (!shape.is_static() || shape.size() != 1) {
            return false;
        }
        const auto size = static_cast<size_t>(shape[0].get_length());
        if (offset == 4 && size == 1 && is_padding_formula_value(input, data, factor_t)) {
            found_temporal_padding = true;
        } else {
            const auto values = get_i64_values(input);
            if (!values || values->size() != size || std::any_of(values->begin(), values->end(), [](int64_t value) {
                    return value != 0;
                })) {
                return false;
            }
        }
        offset += size;
    }
    return offset == 6 && found_temporal_padding;
}

bool is_dynamic_temporal_padding(const ov::Output<ov::Node>& output, const ov::Output<ov::Node>& data, int64_t factor_t) {
    const auto pads_begin = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr());
    if (!pads_begin || pads_begin->get_axis() != 0 || pads_begin->get_input_size() != 2 || !has_values(pads_begin->input_value(0), {0, 0})) {
        return false;
    }

    const auto reverse = ov::as_type_ptr<ov::op::v8::Gather>(pads_begin->get_input_node_shared_ptr(1));
    if (!reverse || reverse->get_batch_dims() != 0 || !has_values(reverse->input_value(1), {2, 1, 0}) || !has_values(reverse->input_value(2), {0})) {
        return false;
    }

    const auto first_column = reverse->input_value(0).get_node_shared_ptr();
    ov::Output<ov::Node> split_output;
    if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(first_column)) {
        if (reshape->get_special_zero() || !has_values(reshape->input_value(1), {3})) {
            return false;
        }
        split_output = reshape->input_value(0);
    } else if (const auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(first_column)) {
        if (!has_values(squeeze->input_value(1), {1})) {
            return false;
        }
        split_output = squeeze->input_value(0);
    } else {
        return false;
    }

    const auto split = ov::as_type_ptr<ov::op::v1::Split>(split_output.get_node_shared_ptr());
    if (!split || split_output.get_index() != 0 || split->get_num_splits() != 2 || !has_values(split->input_value(1), {1})) {
        return false;
    }

    const auto paired_padding = ov::as_type_ptr<ov::op::v1::Reshape>(split->get_input_node_shared_ptr(0));
    if (!paired_padding || paired_padding->get_special_zero() || !has_values(paired_padding->input_value(1), {-1, 2})) {
        return false;
    }

    const auto padding_values = ov::as_type_ptr<ov::op::v0::Concat>(paired_padding->get_input_node_shared_ptr(0));
    return validate_padding_values(padding_values, data, factor_t);
}

bool validate_temporal_padding(const std::shared_ptr<ov::op::v12::Pad>& pad, int64_t factor_t) {
    if (pad->get_pad_mode() != ov::op::PadMode::CONSTANT || !has_vector_size(pad->input_value(1), 5) || !has_values(pad->input_value(2), {0, 0, 0, 0, 0}) ||
        !is_zero_fill(pad->input_value(3))) {
        return false;
    }

    if (const auto pads_begin = get_i64_values(pad->input_value(1))) {
        const auto input_shape = pad->get_input_partial_shape(0);
        if (input_shape.rank() != 5 || !input_shape[2].is_static()) {
            return factor_t == 1 && *pads_begin == std::vector<int64_t>({0, 0, 0, 0, 0});
        }
        const int64_t time = input_shape[2].get_length();
        const int64_t pad_begin_t = (factor_t - time % factor_t) % factor_t;
        return *pads_begin == std::vector<int64_t>({0, 0, pad_begin_t, 0, 0});
    }

    return is_dynamic_temporal_padding(pad->input_value(1), pad->input_value(0), factor_t);
}

bool validate_shape_contract(const std::shared_ptr<ov::op::v1::Reshape>& factor_reshape,
                             const std::shared_ptr<ov::op::v1::Reshape>& flatten_reshape,
                             const std::shared_ptr<ov::op::v1::Reshape>& group_reshape,
                             int64_t& factor_t,
                             int64_t& factor_s,
                             int64_t& output_channels,
                             int64_t& group_size) {
    const auto factor_shape = ov::as_type_ptr<ov::op::v0::Concat>(factor_reshape->get_input_node_shared_ptr(1));
    const auto flatten_shape = ov::as_type_ptr<ov::op::v0::Concat>(flatten_reshape->get_input_node_shared_ptr(1));
    const auto group_shape = ov::as_type_ptr<ov::op::v0::Concat>(group_reshape->get_input_node_shared_ptr(1));
    if (!factor_shape || !flatten_shape || !group_shape || factor_shape->get_axis() != 0 || flatten_shape->get_axis() != 0 || group_shape->get_axis() != 0 ||
        factor_shape->get_input_size() != 7 || group_shape->get_input_size() != 6 || !has_vector_size(factor_shape->input_value(0), 2)) {
        return false;
    }

    const auto factor_t_values = get_i64_values(factor_shape->input_value(2));
    const auto factor_s_values = get_i64_values(factor_shape->input_value(4));
    const auto output_channel_values = get_i64_values(group_shape->input_value(1));
    const auto group_values = get_i64_values(group_shape->input_value(2));
    if (!factor_t_values || factor_t_values->size() != 1 || !factor_s_values || factor_s_values->size() != 1 || !output_channel_values ||
        output_channel_values->size() != 1 || !group_values || group_values->size() != 1 || !has_values(factor_shape->input_value(6), *factor_s_values)) {
        return false;
    }

    factor_t = factor_t_values->front();
    factor_s = factor_s_values->front();
    output_channels = output_channel_values->front();
    group_size = group_values->front();
    if (factor_t <= 0 || factor_s <= 0 || output_channels <= 0 || group_size <= 0) {
        return false;
    }

    const int64_t factor_volume = factor_t * factor_s * factor_s;
    const size_t flatten_spatial_offset = factor_volume == 1 ? 1 : 2;
    const size_t expected_flatten_inputs = factor_volume == 1 ? 4 : 5;
    if (flatten_shape->get_input_size() != expected_flatten_inputs) {
        return false;
    }
    for (size_t index = 0; index < 3; ++index) {
        if (!same_source(factor_shape->input_value(index * 2 + 1), flatten_shape->input_value(index + flatten_spatial_offset)) ||
            !same_source(factor_shape->input_value(index * 2 + 1), group_shape->input_value(index + 3))) {
            return false;
        }
    }
    return true;
}

// Static-shape form of validate_shape_contract(). The factors are read from the inferred reshape output
// shapes instead of the target-shape expressions, so it also holds for special_zero Reshapes and for the
// folded chain without the flatten Reshape. The 8-D view (B, C, T', factor_t, H', factor_s, W', factor_s)
// and the 6-D view (B, output_channels, group_size, T', H', W') are both tied to the padded input
// (B, C, T, H, W): with T = T' * factor_t, H = H' * factor_s and W = W' * factor_s, the 8-D Reshape only
// splits T, H and W, and the trailing Reshape only regroups the channels.
bool derive_factors_from_static_shapes(const ov::Output<ov::Node>& padded_input,
                                       const std::shared_ptr<ov::op::v1::Reshape>& factor_reshape,
                                       const std::shared_ptr<ov::op::v1::Reshape>& group_reshape,
                                       int64_t& factor_t,
                                       int64_t& factor_s,
                                       int64_t& output_channels,
                                       int64_t& group_size) {
    const auto& input_shape = padded_input.get_partial_shape();
    const auto& factor_out = factor_reshape->get_output_partial_shape(0);
    const auto& group_out = group_reshape->get_output_partial_shape(0);
    if (input_shape.is_dynamic() || factor_out.is_dynamic() || group_out.is_dynamic() || input_shape.size() != 5 || factor_out.size() != 8 ||
        group_out.size() != 6) {
        return false;
    }

    const auto input = input_shape.to_shape();
    const auto factor = factor_out.to_shape();
    const auto group = group_out.to_shape();
    if (factor[0] != input[0] || factor[1] != input[1] || factor[5] != factor[7] || factor[2] * factor[3] != input[2] || factor[4] * factor[5] != input[3] ||
        factor[6] * factor[7] != input[4]) {
        return false;
    }
    if (group[0] != input[0] || group[3] != factor[2] || group[4] != factor[4] || group[5] != factor[6]) {
        return false;
    }

    factor_t = static_cast<int64_t>(factor[3]);
    factor_s = static_cast<int64_t>(factor[5]);
    output_channels = static_cast<int64_t>(group[1]);
    group_size = static_cast<int64_t>(group[2]);
    return factor_t > 0 && factor_s > 0 && output_channels > 0 && group_size > 0;
}

// The mean runs over the group axis of the 6-D view (B, output_channels, group_size, T', H', W'), i.e. axis 2.
// The equivalent negative axis -4 is accepted as well.
bool reduces_group_axis(const ov::Output<ov::Node>& output) {
    const auto& node = output.get_node_shared_ptr();
    const auto axes = get_i64_values(node->input_value(1));
    const auto rank = node->get_input_partial_shape(0).rank();
    if (!axes || axes->size() != 1 || rank.is_dynamic() || rank.get_length() != 6) {
        return false;
    }
    const int64_t axis = axes->front();
    return axis == 2 || axis == 2 - rank.get_length();
}

}  // namespace

FuseAvgDown::FuseAvgDown() {
    using namespace ov::pass::pattern;

    // The temporal Pad is validated in the callback (its pads may come from a ShapeOf-based expression).
    auto pad_m = optional<ov::op::v12::Pad>({any_input(), any_input(), any_input(), any_input()});
    auto factor_reshape_m = wrap_type<ov::op::v1::Reshape>({pad_m, any_input()}, consumers_count(1));
    auto transpose_order_m = wrap_type<ov::op::v0::Constant>(value_matches("0, 1, 3, 5, 7, 2, 4, 6"));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({factor_reshape_m, transpose_order_m}, consumers_count(1));
    // The flatten Reshape is optional: earlier graph optimizations usually merge it with the group Reshape.
    auto flatten_reshape_m = optional<ov::op::v1::Reshape>({transpose_m, any_input()}, consumers_count(1));
    auto group_reshape_m = wrap_type<ov::op::v1::Reshape>({flatten_reshape_m, any_input()}, consumers_count(1));
    auto reduce_m = wrap_type<ov::op::v1::ReduceMean>({group_reshape_m, wrap_const()}, reduces_group_axis, {{"keep_dims", false}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(pattern_map.at(reduce_m).get_node_shared_ptr());
        if (transformation_callback(reduce)) {
            return false;
        }
        const auto factor_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(factor_reshape_m).get_node_shared_ptr());
        const auto transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
        const auto group_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(group_reshape_m).get_node_shared_ptr());
        std::shared_ptr<ov::op::v1::Reshape> flatten_reshape;
        if (pattern_map.count(flatten_reshape_m)) {
            flatten_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(flatten_reshape_m).get_node_shared_ptr());
        }
        std::shared_ptr<ov::op::v12::Pad> pad;
        if (pattern_map.count(pad_m)) {
            pad = ov::as_type_ptr<ov::op::v12::Pad>(pattern_map.at(pad_m).get_node_shared_ptr());
        }
        OPENVINO_ASSERT(reduce && factor_reshape && transpose && group_reshape, "FuseAvgDown: unexpected node types in the matched pattern");

        // Static shapes: the factors come from the inferred reshape outputs, which is independent of special_zero.
        // Dynamic shapes: the factors come from the target-shape expressions; with special_zero a literal 0 in them
        // means "copy this dimension" rather than a length, so such a chain is left untouched.
        const auto padded_input = factor_reshape->input_value(0);
        int64_t factor_t = 0;
        int64_t factor_s = 0;
        int64_t output_channels = 0;
        int64_t group_size = 0;
        if (!derive_factors_from_static_shapes(padded_input, factor_reshape, group_reshape, factor_t, factor_s, output_channels, group_size)) {
            const bool special_zero =
                factor_reshape->get_special_zero() || group_reshape->get_special_zero() || (flatten_reshape && flatten_reshape->get_special_zero());
            if (special_zero || !flatten_reshape ||
                !validate_shape_contract(factor_reshape, flatten_reshape, group_reshape, factor_t, factor_s, output_channels, group_size)) {
                return false;
            }
        }

        ov::NodeVector matched_nodes = {factor_reshape, transpose, group_reshape, reduce};
        if (flatten_reshape) {
            matched_nodes.push_back(flatten_reshape);
        }
        ov::Output<ov::Node> unpadded_input = padded_input;
        if (pad) {
            if (!validate_temporal_padding(pad, factor_t)) {
                return false;
            }
            unpadded_input = pad->input_value(0);
            matched_nodes.push_back(pad);
        }

        const auto input_shape = unpadded_input.get_partial_shape();
        if (input_shape.rank() != 5 || !input_shape[1].is_static()) {
            return false;
        }
        const int64_t input_channels = input_shape[1].get_length();
        const int64_t factor_volume = factor_t * factor_s * factor_s;
        if (input_channels * factor_volume != output_channels * group_size) {
            return false;
        }

        if (factor_t == 1 && factor_s == 1 && output_channels == input_channels && group_size == 1) {
            if (!input_shape.compatible(reduce->get_output_partial_shape(0))) {
                return false;
            }
            return ov::replace_output_update_name(reduce->output(0), unpadded_input);
        }

        std::shared_ptr<ov::Node> fused;
        if (factor_t == 2 && factor_s == 2 && group_size == 4) {
            fused = std::make_shared<ov::intel_gpu::op::GroupedSpaceToDepth>(unpadded_input,
                                                                             static_cast<size_t>(factor_t),
                                                                             static_cast<size_t>(factor_s),
                                                                             static_cast<size_t>(output_channels));
        } else if (factor_t == 1 && factor_s == 2 && output_channels == input_channels && group_size == 4) {
            fused = std::make_shared<ov::op::v1::AvgPool>(unpadded_input,
                                                          ov::Strides{1, 2, 2},
                                                          ov::Shape{0, 0, 0},
                                                          ov::Shape{0, 0, 0},
                                                          ov::Shape{1, 2, 2},
                                                          false,
                                                          ov::op::RoundingType::FLOOR,
                                                          ov::op::PadType::EXPLICIT);
        } else {
            return false;
        }
        if (!fused->get_output_partial_shape(0).compatible(reduce->get_output_partial_shape(0))) {
            return false;
        }

        fused->set_friendly_name(reduce->get_friendly_name());
        ov::copy_runtime_info(matched_nodes, fused);
        ov::replace_node(reduce, fused);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(reduce_m, "FuseAvgDown");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
