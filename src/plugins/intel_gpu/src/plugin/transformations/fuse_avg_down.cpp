// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_avg_down.hpp"

#include <optional>
#include <unordered_set>

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
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
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

bool contains_padding_formula(const ov::Output<ov::Node>& output,
                              const ov::Output<ov::Node>& data,
                              int64_t factor_t,
                              std::unordered_set<const ov::Node*>& visited) {
    const auto node = output.get_node_shared_ptr();
    if (!visited.insert(node.get()).second) {
        return false;
    }
    if (is_padding_formula(node, data, factor_t)) {
        return true;
    }
    if (ov::is_type<ov::op::v3::ShapeOf>(node)) {
        return false;
    }
    for (const auto& input : node->inputs()) {
        if (contains_padding_formula(input.get_source_output(), data, factor_t, visited)) {
            return true;
        }
    }
    return false;
}

bool validate_temporal_padding(const std::shared_ptr<ov::op::v12::Pad>& pad, int64_t factor_t) {
    if (pad->get_pad_mode() != ov::op::PadMode::CONSTANT || !has_vector_size(pad->input_value(1), 5) || !has_values(pad->input_value(2), {0, 0, 0, 0, 0}) ||
        !has_values(pad->input_value(3), {0})) {
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

    std::unordered_set<const ov::Node*> visited;
    return contains_padding_formula(pad->input_value(1), pad->input_value(0), factor_t, visited);
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

}  // namespace

FuseAvgDown::FuseAvgDown() {
    using namespace ov::pass::pattern;

    auto input_m = any_input();
    auto factor_reshape_m = wrap_type<ov::op::v1::Reshape>({input_m, any_input()}, consumers_count(1));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({factor_reshape_m, wrap_type<ov::op::v0::Constant>()}, consumers_count(1));
    auto flatten_reshape_m = wrap_type<ov::op::v1::Reshape>({transpose_m, any_input()}, consumers_count(1));
    auto group_reshape_m = wrap_type<ov::op::v1::Reshape>({flatten_reshape_m, any_input()}, consumers_count(1));
    auto reduce_m = wrap_type<ov::op::v1::ReduceMean>({group_reshape_m, wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto input = pattern_map.at(input_m);
        const auto factor_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(factor_reshape_m).get_node_shared_ptr());
        const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_m).get_node_shared_ptr());
        const auto flatten_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(flatten_reshape_m).get_node_shared_ptr());
        const auto group_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(group_reshape_m).get_node_shared_ptr());
        const auto reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(pattern_map.at(reduce_m).get_node_shared_ptr());

        if (transformation_callback(reduce) || factor_reshape->get_special_zero() || flatten_reshape->get_special_zero() || group_reshape->get_special_zero() ||
            reduce->get_keep_dims() || !has_values(reduce->input_value(1), {2}) || !has_values(transpose->input_value(1), {0, 1, 3, 5, 7, 2, 4, 6})) {
            return false;
        }

        int64_t factor_t = 0;
        int64_t factor_s = 0;
        int64_t output_channels = 0;
        int64_t group_size = 0;
        if (!validate_shape_contract(factor_reshape, flatten_reshape, group_reshape, factor_t, factor_s, output_channels, group_size)) {
            return false;
        }

        ov::NodeVector matched_nodes = {factor_reshape, transpose, flatten_reshape, group_reshape, reduce};
        std::shared_ptr<ov::op::v12::Pad> pad;
        ov::Output<ov::Node> unpadded_input = input;
        if ((pad = ov::as_type_ptr<ov::op::v12::Pad>(input.get_node_shared_ptr()))) {
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
            return ov::replace_output_update_name(reduce->output(0), unpadded_input);
        }

        if (factor_t == 2 && factor_s == 2 && group_size == 4) {
            auto grouped_space_to_depth = std::make_shared<ov::intel_gpu::op::GroupedSpaceToDepth>(unpadded_input,
                                                                                                   static_cast<size_t>(factor_t),
                                                                                                   static_cast<size_t>(factor_s),
                                                                                                   static_cast<size_t>(output_channels));
            if (!grouped_space_to_depth->get_output_partial_shape(0).compatible(reduce->get_output_partial_shape(0))) {
                return false;
            }

            grouped_space_to_depth->set_friendly_name(reduce->get_friendly_name());
            ov::copy_runtime_info(matched_nodes, grouped_space_to_depth);
            ov::replace_node(reduce, grouped_space_to_depth);
            return true;
        }

        if (factor_t != 1 || factor_s != 2 || output_channels != input_channels || group_size != 4) {
            return false;
        }

        auto pool = std::make_shared<ov::op::v1::AvgPool>(unpadded_input,
                                                          ov::Strides{1, 2, 2},
                                                          ov::Shape{0, 0, 0},
                                                          ov::Shape{0, 0, 0},
                                                          ov::Shape{1, 2, 2},
                                                          false,
                                                          ov::op::RoundingType::FLOOR,
                                                          ov::op::PadType::EXPLICIT);
        if (!pool->get_output_partial_shape(0).compatible(reduce->get_output_partial_shape(0))) {
            return false;
        }

        pool->set_friendly_name(reduce->get_friendly_name());
        ov::copy_runtime_info(matched_nodes, pool);
        ov::replace_node(reduce, pool);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(reduce_m, "FuseAvgDown");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
