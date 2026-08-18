// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/move_eltwise_up_data_movement.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v7 = ov::op::v7;
namespace op_util = ov::op::util;
namespace {
bool is_data_movement_operation(const std::shared_ptr<ov::Node>& node,
                                const std::vector<ov::DiscreteTypeInfo>& allowed_data_movement_ops) {
    return std::any_of(allowed_data_movement_ops.begin(), allowed_data_movement_ops.end(), [&](const auto& type) {
        return node->get_type_info().is_castable(type);
    });
}
}  // namespace

std::vector<ov::DiscreteTypeInfo> ov::pass::MoveEltwiseUpThroughDataMov::get_default_allowed_ops() {
    return {
        v0::Squeeze::get_type_info_static(),
        v0::Unsqueeze::get_type_info_static(),
        v1::Reshape::get_type_info_static(),
        v1::Transpose::get_type_info_static(),
        v0::ShuffleChannels::get_type_info_static(),
        v7::Roll::get_type_info_static(),
        v0::ReverseSequence::get_type_info_static(),
        v0::DepthToSpace::get_type_info_static(),
        v1::BatchToSpace::get_type_info_static(),
        op_util::BroadcastBase::get_type_info_static(),
        op_util::GatherBase::get_type_info_static(),
    };
}

ov::pass::MoveEltwiseUpThroughDataMovScalar::MoveEltwiseUpThroughDataMovScalar(
    std::vector<DiscreteTypeInfo> allowed_data_movement_ops) {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMovScalar);
    auto eltwise_pattern =
        wrap_type<op_util::UnaryElementwiseArithmetic, op_util::BinaryElementwiseArithmetic, v0::FakeQuantize>(
            ov::pass::pattern::has_static_rank());

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        if (transformation_callback(eltwise)) {
            return false;
        }

        if (eltwise->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        for (size_t i = 1; i < eltwise->get_input_size(); ++i) {
            if (!ov::op::util::is_scalar_or_single_elem_constant(
                    ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(i)))) {
                return false;
            }
        }

        auto current = eltwise->get_input_node_shared_ptr(0);
        auto child = eltwise;

        while (is_data_movement_operation(current, allowed_data_movement_ops)) {
            if (current->get_output_size() != 1 || current->get_output_target_inputs(0).size() != 1 ||
                current->get_output_element_type(0) != current->get_input_element_type(0)) {
                return false;
            }

            child = current;
            current = current->get_input_node_shared_ptr(0);
        }

        // now current is the first not data movement op
        if (child == eltwise) {
            return false;
        }

        // eltwise constant shape should match new input shape
        for (size_t i = 1; i < eltwise->get_input_size(); i++) {
            if (current->get_output_partial_shape(0).size() != eltwise->get_input_partial_shape(i).size()) {
                auto old_eltwise_const = ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(i));
                if (old_eltwise_const->get_shape().size() != 0) {
                    auto new_constant = std::make_shared<v0::Constant>(*old_eltwise_const.get(), ov::Shape{});
                    ov::copy_runtime_info(old_eltwise_const, new_constant);
                    eltwise->input(i).replace_source_output(new_constant->output(0));
                }
            }
        }
        ov::replace_output_update_name(eltwise->output(0), eltwise->input_value(0));

        ov::OutputVector eltwise_inputs = eltwise->input_values();
        eltwise_inputs[0] = child->input_value(0);
        auto new_eltwise = eltwise->clone_with_new_inputs(eltwise_inputs);
        // WA: it's necessary to set empty friendly name here
        // to avoid name duplication in TypeRelaxed cases
        new_eltwise->set_friendly_name("");
        ov::copy_runtime_info(eltwise, new_eltwise);

        ov::OutputVector child_inputs = child->input_values();
        child_inputs[0] = new_eltwise;
        auto new_child = child->clone_with_new_inputs(child_inputs);
        ov::copy_runtime_info(child, new_child);
        new_child->set_friendly_name(child->get_friendly_name());

        ov::replace_node(child, new_child);
        return true;
    };

    auto m = std::make_shared<Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::MoveEltwiseUpThroughDataMovPerChannel::MoveEltwiseUpThroughDataMovPerChannel() {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMovPerChannel);

    auto const_predicate = [](const ov::Output<ov::Node>& output) {
        auto constant_op = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
        if (!constant_op)
            return false;

        if (output.get_target_inputs().size() != 1)
            return false;

        const auto& shape = constant_op->get_shape();
        return std::count_if(shape.begin(), shape.end(), [](size_t v) {
                   return v > 1;
               }) == 1;
    };

    auto eltw_predicate = [](const ov::Output<ov::Node>& output) {
        if (output.get_target_inputs().size() != 1)
            return false;

        auto node = output.get_node();

        if (node->get_output_partial_shape(0).rank().is_dynamic())
            return false;

        const size_t const_idx = ov::is_type<v0::Constant>(node->get_input_node_ptr(0)) ? 0 : 1;
        const size_t data_flow_idx = (const_idx + 1) % 2;

        if (node->get_input_partial_shape(data_flow_idx).size() < node->get_input_partial_shape(const_idx).size())
            return false;

        return true;
    };

    auto eltw_data_flow_in = wrap_type<v1::Reshape, v0::Squeeze, v0::Unsqueeze>(ov::pass::pattern::consumers_count(1));
    auto eltw_const_in = wrap_type<v0::Constant>(const_predicate);
    auto eltwise_pattern =
        wrap_type<op_util::BinaryElementwiseArithmetic>({eltw_data_flow_in, eltw_const_in}, eltw_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        if (transformation_callback(eltwise)) {
            return false;
        }

        auto binary_eltwise = ov::as_type_ptr<op_util::BinaryElementwiseArithmetic>(eltwise);
        if (!binary_eltwise || binary_eltwise->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }

        const size_t const_idx = ov::is_type<v0::Constant>(eltwise->get_input_node_ptr(0)) ? 0 : 1;
        const size_t data_flow_idx = (const_idx + 1) % 2;

        auto const_shape = eltwise->get_input_shape(const_idx);
        size_t channel_idx = 0;
        size_t channel_val = 0;
        for (size_t i = 0; i < const_shape.size(); i++) {
            if (const_shape[i] > 1) {
                channel_idx = i;
                channel_val = const_shape[i];
            }
        }

        auto parent = eltwise->get_input_node_shared_ptr(data_flow_idx);
        const auto& parent_in_pshape = parent->get_input_partial_shape(0);
        const auto& parent_out_pshape = parent->get_output_partial_shape(0);
        if (parent_in_pshape.is_dynamic() || parent_out_pshape.is_dynamic())
            return false;

        const size_t in_rank = parent_in_pshape.size();
        const size_t out_rank = parent_out_pshape.size();
        const size_t const_rank = const_shape.size();
        if (const_rank > out_rank)
            return false;

        // The constant is right-aligned against the data flow shape by NumPy broadcasting, so its
        // non-unit axis sits at this position in the parent's output space.
        const size_t output_channel_idx = out_rank - const_rank + channel_idx;
        if (static_cast<size_t>(parent_out_pshape[output_channel_idx].get_length()) != channel_val)
            return false;

        const auto trailing_stride = [](const ov::PartialShape& pshape, size_t axis) -> std::optional<int64_t> {
            int64_t stride = 1;
            for (size_t i = axis + 1; i < pshape.size(); ++i) {
                const int64_t dim = pshape[i].get_length();
                if (dim != 0 && stride > std::numeric_limits<int64_t>::max() / dim)
                    return std::nullopt;
                stride *= dim;
            }
            return stride;
        };

        const auto output_stride = trailing_stride(parent_out_pshape, output_channel_idx);
        if (!output_stride.has_value())
            return false;

        // Reshape/Squeeze/Unsqueeze preserve element order, so the output axis corresponds to the
        // input axis that has both the same extent and the same trailing stride. Anything ambiguous
        // is rejected rather than guessed: matching on extent alone picks the wrong axis whenever
        // two axes happen to share a size.
        size_t input_channel_idx = in_rank;
        for (size_t i = 0; i < in_rank; ++i) {
            if (static_cast<size_t>(parent_in_pshape[i].get_length()) != channel_val)
                continue;
            const auto input_stride = trailing_stride(parent_in_pshape, i);
            if (!input_stride.has_value() || input_stride != output_stride)
                continue;
            if (input_channel_idx != in_rank)
                return false;
            input_channel_idx = i;
        }
        if (input_channel_idx == in_rank)
            return false;

        auto new_shape = ov::Shape(in_rank, 1);

        new_shape[input_channel_idx] = channel_val;
        auto old_const = ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(const_idx));
        auto new_const = std::make_shared<v0::Constant>(*old_const, new_shape);
        ov::replace_node_update_name(old_const, new_const);
        ov::replace_output_update_name(eltwise->output(0), eltwise->input_value(data_flow_idx));

        ov::OutputVector eltwise_inputs = eltwise->input_values();
        eltwise_inputs[data_flow_idx] = parent->input_value(0);
        auto new_eltwise = eltwise->clone_with_new_inputs(eltwise_inputs);
        ov::copy_runtime_info(eltwise, new_eltwise);

        ov::OutputVector parent_inputs = parent->input_values();
        parent_inputs[0] = new_eltwise;
        auto new_parent = parent->clone_with_new_inputs(parent_inputs);
        ov::copy_runtime_info(parent, new_parent);
        new_parent->set_friendly_name(parent->get_friendly_name());

        ov::replace_node(parent, new_parent);
        return true;
    };

    auto m = std::make_shared<Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}
