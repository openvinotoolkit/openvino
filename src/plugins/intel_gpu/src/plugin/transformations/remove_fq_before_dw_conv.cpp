// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_fq_before_dw_conv.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

namespace {

constexpr size_t channel_axis = 1;
constexpr size_t max_bridge_operations = 2;
constexpr size_t max_depthwise_kernel_elements = 9;

bool has_supported_depthwise_shape(const std::shared_ptr<ov::op::v1::GroupConvolution>& convolution) {
    const auto& data_shape = convolution->get_input_partial_shape(0);
    const auto& weights_shape = convolution->get_input_partial_shape(1);
    const auto& output_shape = convolution->get_output_partial_shape(0);
    if (!data_shape[channel_axis].is_static() || !weights_shape.is_static() || !output_shape[channel_axis].is_static()) {
        return false;
    }

    if (data_shape.size() != 4 || weights_shape.size() != 5 || output_shape.size() != 4) {
        return false;
    }

    const size_t groups = weights_shape[0].get_length();
    if (groups <= 1 || weights_shape[1] != 1 || weights_shape[2] != 1 || data_shape[channel_axis] != groups || output_shape[channel_axis] != groups) {
        return false;
    }

    const size_t kernel_y = weights_shape[3].get_length();
    const size_t kernel_x = weights_shape[4].get_length();
    return kernel_y > 0 && kernel_x > 0 && kernel_y <= max_depthwise_kernel_elements && kernel_x <= max_depthwise_kernel_elements &&
           kernel_y * kernel_x <= max_depthwise_kernel_elements;
}

bool has_channelwise_fake_quantize(const std::shared_ptr<ov::op::v0::FakeQuantize>& fake_quantize, const std::shared_ptr<ov::op::v1::GroupConvolution>& convolution) {
    if (convolution->get_input_partial_shape(0)[channel_axis].is_dynamic()) {
        return false;
    }
    size_t channels = convolution->get_input_partial_shape(0)[channel_axis].get_length();

    const ov::Shape expected_shape{1, channels, 1, 1};
    for (size_t index = 1; index < fake_quantize->get_input_size(); ++index) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(fake_quantize->get_input_node_shared_ptr(index));
        if (constant == nullptr || constant->get_shape() != expected_shape) {
            return false;
        }
    }

    return true;
}

bool has_channelwise_int8_weights(const std::shared_ptr<ov::op::v1::GroupConvolution>& convolution) {
    const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(convolution->get_input_node_shared_ptr(1));
    if (multiply == nullptr) {
        return false;
    }

    const auto weights_shape = convolution->get_input_partial_shape(1);
    const ov::PartialShape expected_scale_shape{weights_shape[0], 1, 1, 1, 1};
    for (size_t scale_index = 0; scale_index < 2; ++scale_index) {
        const auto scale = ov::as_type_ptr<ov::op::v0::Constant>(multiply->get_input_node_shared_ptr(scale_index));
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(multiply->get_input_node_shared_ptr(1 - scale_index));
        if (scale == nullptr || scale->get_output_partial_shape(0) != expected_scale_shape || convert == nullptr) {
            continue;
        }

        const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
        if (weights == nullptr || weights->get_output_partial_shape(0) != weights_shape) {
            continue;
        }

        const auto weights_type = weights->get_element_type();
        if (weights_type == ov::element::i8 || weights_type == ov::element::u8) {
            return true;
        }
    }

    return false;
}

std::optional<size_t> get_transposed_channel_axis(const std::shared_ptr<ov::op::v1::Transpose>& transpose, size_t current_channel_axis) {
    const auto rank = transpose->get_input_partial_shape(0).rank();
    if (rank.is_dynamic() || rank.get_length() <= 0 || static_cast<uint64_t>(rank.get_length()) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::nullopt;
    }

    const auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
    if (order == nullptr) {
        return std::nullopt;
    }

    const int64_t rank_value = rank.get_length();
    const auto order_values = order->cast_vector<int64_t>();
    if (order_values.size() != static_cast<size_t>(rank_value) || current_channel_axis >= order_values.size()) {
        return std::nullopt;
    }

    std::vector<bool> visited_axes(order_values.size(), false);
    std::optional<size_t> new_channel_axis;
    for (size_t output_axis = 0; output_axis < order_values.size(); ++output_axis) {
        int64_t input_axis = order_values[output_axis];
        if (input_axis < 0) {
            input_axis += rank_value;
        }
        if (input_axis < 0 || input_axis >= rank_value || visited_axes[static_cast<size_t>(input_axis)]) {
            return std::nullopt;
        }

        visited_axes[static_cast<size_t>(input_axis)] = true;
        if (static_cast<size_t>(input_axis) == current_channel_axis) {
            new_channel_axis = output_axis;
        }
    }

    return new_channel_axis;
}

bool mvn_reduces_channel(const std::shared_ptr<ov::Node>& node, size_t current_channel_axis) {
    const auto rank = node->get_input_partial_shape(0).rank();
    if (rank.is_dynamic() || rank.get_length() <= 0 || static_cast<uint64_t>(rank.get_length()) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        current_channel_axis >= static_cast<size_t>(rank.get_length())) {
        return false;
    }

    if (const auto mvn = ov::as_type_ptr<ov::op::v0::MVN>(node)) {
        return mvn->get_reduction_axes().count(current_channel_axis) != 0;
    }

    const auto mvn = ov::as_type_ptr<ov::op::v6::MVN>(node);
    if (mvn == nullptr) {
        return false;
    }

    const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(mvn->get_input_node_shared_ptr(1));
    if (axes == nullptr) {
        return false;
    }

    const int64_t rank_value = rank.get_length();
    for (int64_t axis : axes->cast_vector<int64_t>()) {
        if (axis < 0) {
            axis += rank_value;
        }
        if (axis < 0 || axis >= rank_value) {
            return false;
        }
        if (static_cast<size_t>(axis) == current_channel_axis) {
            return true;
        }
    }

    return false;
}

bool has_mvn_dequantization_barrier(const ov::Output<ov::Node>& output, size_t current_channel_axis, size_t bridge_operations = 0) {
    for (const auto& target_input : output.get_target_inputs()) {
        const auto consumer = target_input.get_node()->shared_from_this();
        if (target_input.get_index() == 0 && mvn_reduces_channel(consumer, current_channel_axis)) {
            return true;
        }

        if (bridge_operations >= max_bridge_operations) {
            continue;
        }

        if (const auto add = ov::as_type_ptr<ov::op::v1::Add>(consumer)) {
            const size_t data_index = target_input.get_index();
            if (data_index >= add->get_input_size()) {
                continue;
            }

            const size_t other_index = 1 - data_index;
            if (ov::is_type<ov::op::v0::Constant>(add->get_input_node_shared_ptr(other_index)) &&
                has_mvn_dequantization_barrier(add->output(0), current_channel_axis, bridge_operations + 1)) {
                return true;
            }
        } else if (const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(consumer)) {
            if (target_input.get_index() != 0) {
                continue;
            }

            const auto transposed_channel_axis = get_transposed_channel_axis(transpose, current_channel_axis);
            if (transposed_channel_axis.has_value() && has_mvn_dequantization_barrier(transpose->output(0), *transposed_channel_axis, bridge_operations + 1)) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace

RemoveFakeQuantizeBeforeDepthwiseConv::RemoveFakeQuantizeBeforeDepthwiseConv() {
    auto group_convolution_pattern = ov::pass::pattern::wrap_type<ov::op::v1::GroupConvolution>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& matcher) {
        const auto group_convolution = ov::as_type_ptr<ov::op::v1::GroupConvolution>(matcher.get_match_root());
        if (group_convolution == nullptr || !has_supported_depthwise_shape(group_convolution)) {
            return false;
        }

        const auto fake_quantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(group_convolution->get_input_node_shared_ptr(0));
        if (fake_quantize == nullptr || !has_channelwise_fake_quantize(fake_quantize, group_convolution) ||
            !has_channelwise_int8_weights(group_convolution)) {
            return false;
        }

        const auto fq_consumers = fake_quantize->output(0).get_target_inputs();
        if (fq_consumers.size() != 1) {
            return false;
        }

        const auto& fq_consumer = *fq_consumers.begin();
        if (fq_consumer.get_node() != group_convolution.get() || fq_consumer.get_index() != 0) {
            return false;
        }

        const auto fq_data = fake_quantize->input_value(0);
        if (fq_data.get_target_inputs().size() < 2 || !has_mvn_dequantization_barrier(group_convolution->output(0), channel_axis)) {
            return false;
        }

        group_convolution->input(0).replace_source_output(fq_data);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(group_convolution_pattern, "RemoveFakeQuantizeBeforeDepthwiseConv");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
