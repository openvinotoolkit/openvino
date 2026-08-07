// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_eltwise_up_data_movement_per_channel.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

}  // namespace

ov::intel_gpu::MoveEltwiseUpThroughDataMovPerChannel::MoveEltwiseUpThroughDataMovPerChannel() {
    auto const_predicate = [](const ov::Output<ov::Node>& output) {
        auto constant_op = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
        if (!constant_op)
            return false;

        if (output.get_target_inputs().size() != 1)
            return false;

        const auto& shape = constant_op->get_shape();
        return std::count_if(shape.begin(), shape.end(), [](size_t value) {
                   return value > 1;
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

        return node->get_input_partial_shape(data_flow_idx).size() >= node->get_input_partial_shape(const_idx).size();
    };

    auto eltw_data_flow_in = wrap_type<v1::Reshape, v0::Squeeze, v0::Unsqueeze>(ov::pass::pattern::consumers_count(1));
    auto eltw_const_in = wrap_type<v0::Constant>(const_predicate);
    auto eltwise_pattern = wrap_type<op_util::BinaryElementwiseArithmetic>({eltw_data_flow_in, eltw_const_in}, eltw_predicate);

    ov::matcher_pass_callback callback = [this, eltwise_pattern](Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        if (transformation_callback(eltwise))
            return false;

        auto binary_eltwise = std::dynamic_pointer_cast<op_util::BinaryElementwiseArithmetic>(eltwise);
        if (!binary_eltwise || binary_eltwise->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY)
            return false;

        const size_t const_idx = ov::is_type<v0::Constant>(eltwise->get_input_node_ptr(0)) ? 0 : 1;
        const size_t data_flow_idx = (const_idx + 1) % 2;

        const auto const_shape = eltwise->get_input_shape(const_idx);
        size_t channel_idx = 0;
        size_t channel_val = 0;
        for (size_t index = 0; index < const_shape.size(); ++index) {
            if (const_shape[index] > 1) {
                channel_idx = index;
                channel_val = const_shape[index];
            }
        }

        auto parent = eltwise->get_input_node_shared_ptr(data_flow_idx);
        const auto& parent_in_pshape = parent->get_input_partial_shape(0);
        const auto& parent_out_pshape = parent->get_output_partial_shape(0);

        if (parent_in_pshape.rank().is_dynamic() || parent_out_pshape.rank().is_dynamic())
            return false;

        const size_t in_rank = parent_in_pshape.size();
        const size_t out_rank = parent_out_pshape.size();

        for (const auto& dimension : parent_in_pshape) {
            if (dimension.is_dynamic())
                return false;
        }
        for (const auto& dimension : parent_out_pshape) {
            if (dimension.is_dynamic())
                return false;
        }

        const size_t const_rank = const_shape.size();
        if (const_rank > out_rank)
            return false;

        const size_t output_channel_idx = out_rank - const_rank + channel_idx;
        if (output_channel_idx >= out_rank || static_cast<size_t>(parent_out_pshape[output_channel_idx].get_length()) != channel_val) {
            return false;
        }

        int64_t output_stride = 1;
        for (size_t index = output_channel_idx + 1; index < out_rank; ++index) {
            const int64_t dimension = parent_out_pshape[index].get_length();
            if (dimension > 0 && output_stride > std::numeric_limits<int64_t>::max() / dimension)
                return false;
            output_stride *= dimension;
        }

        size_t input_channel_idx = in_rank;
        for (size_t index = 0; index < in_rank; ++index) {
            if (static_cast<size_t>(parent_in_pshape[index].get_length()) != channel_val)
                continue;

            int64_t input_stride = 1;
            bool overflow = false;
            for (size_t trailing_index = index + 1; trailing_index < in_rank; ++trailing_index) {
                const int64_t dimension = parent_in_pshape[trailing_index].get_length();
                if (dimension > 0 && input_stride > std::numeric_limits<int64_t>::max() / dimension) {
                    overflow = true;
                    break;
                }
                input_stride *= dimension;
            }

            if (overflow)
                continue;

            if (input_stride == output_stride) {
                if (input_channel_idx != in_rank)
                    return false;
                input_channel_idx = index;
            }
        }

        if (input_channel_idx == in_rank)
            return false;

        auto new_shape = ov::Shape(in_rank, 1);
        new_shape[input_channel_idx] = channel_val;
        auto old_constant = ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(const_idx));
        auto new_constant = std::make_shared<v0::Constant>(*old_constant, new_shape);
        ov::replace_node_update_name(old_constant, new_constant);
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

    auto matcher = std::make_shared<Matcher>(eltwise_pattern, "MoveEltwiseUpThroughDataMovPerChannelGPU");
    register_matcher(matcher, callback);
}
