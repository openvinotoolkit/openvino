// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_conv_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::MultiplyConvolutionFusion::MultiplyConvolutionFusion() {
    MATCHER_SCOPE(MultiplyConvolutionFusion);
    auto input_pattern = pattern::any_input(pattern::has_static_rank());
    auto mul_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input_pattern, mul_const_pattern},
                                                                          pattern::consumers_count(1));
    auto weights_pattern = pass::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // Can't fuse Multiply to Convolution if that Multiply is part of dequantization subgraph
        // since that breaks low precision transformations
        if (op::util::is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        const auto& mul_const = pattern_to_output.at(mul_const_pattern);
        const auto& input = pattern_to_output.at(input_pattern);

        auto weights_shape = weights.get_partial_shape();
        auto mul_const_shape = mul_const.get_partial_shape();
        auto input_shape = input.get_partial_shape();

        // Check if constant in multiply broadcasts input's shape.
        // If this is the case, we cannot perform the transformation as
        // 'bare' input's shape will not be aligned with weights.
        if (ov::op::util::check_for_broadcast(mul_const_shape, input_shape)) {
            return false;
        }

        // Check if mul_const is broadcastable to weights.
        // Also if mul_const's rank matches weights rank and mul_const.shape[0] != 1
        // then we can't fuse the multiply, since first dimension in mul_const corresponds to
        // batch size, while first dimension in weights corresponds to output channel count
        if (!ov::op::util::check_for_broadcast(weights_shape, mul_const_shape) ||
            (weights_shape.size() == mul_const_shape.size() && mul_const_shape[0] != 1)) {
            return false;
        }

        auto weights_multiply = std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = ov::util::get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()}, {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::MultiplyGroupConvolutionFusion::MultiplyGroupConvolutionFusion() {
    MATCHER_SCOPE(MultiplyGroupConvolutionFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input_pattern, mul_const_pattern},
                                                                          pattern::consumers_count(1));
    auto weights_pattern = pass::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ov::pass::pattern::wrap_type<ov::op::v1::GroupConvolution>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // Can't fuse Multiply to Convolution if that Multiply is part of dequantization subgraph
        // since that breaks low precision transformations
        if (op::util::is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        std::shared_ptr<Node> mul_const = pattern_to_output.at(mul_const_pattern).get_node_shared_ptr();
        const auto& weights_shape = weights.get_shape();

        const auto& input = pattern_to_output.at(input_pattern);

        auto mul_const_shape = mul_const->output(0).get_partial_shape();
        auto input_shape = input.get_partial_shape();
        // Check if constant in multiply broadcasts input's shape.
        // If this is the case, we cannot perform the transformation as
        // 'bare' input's shape will not be aligned with weights.
        if (ov::op::util::check_for_broadcast(mul_const_shape, input_shape)) {
            return false;
        }

        if (shape_size(mul_const->get_shape()) > 1) {
            auto mul_const_shape = mul_const->get_shape();
            // extend mul_const_shape rank with unit dimensions
            if (weights_shape.size() - mul_const_shape.size() > 1)
                mul_const_shape.insert(mul_const_shape.begin(), weights_shape.size() - mul_const_shape.size() - 1, 1);
            // if mul_const.shape[0] != 1
            // then we can't fuse the multiply, since first dimension in mul_const corresponds to
            // batch size, while first dimension in weights corresponds to output channel count
            if (mul_const_shape[0] != 1)
                return false;
            auto G = mul_const_shape[1] > 1 ? weights_shape[0] : 1;
            auto C = mul_const_shape[1] / G;
            // Reshape mul_const from shape (1, C, H, W) to (G, 1, C / G, H, W) to match GroupConvolution weights format
            Shape new_shape{G, 1, C};
            std::copy(mul_const_shape.begin() + 2, mul_const_shape.end(), std::back_inserter(new_shape));
            if (!ov::op::util::check_for_broadcast(weights_shape, new_shape)) {
                return false;
            }
            mul_const = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(element::u64, Shape{new_shape.size()}, new_shape),
                false);
        }

        auto weights_multiply = std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = ov::util::get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()}, {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::MultiplyConvolutionBackpropDataFusion::MultiplyConvolutionBackpropDataFusion() {
    MATCHER_SCOPE(MultiplyConvolutionBackpropDataFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input_pattern, mul_const_pattern},
                                                                          pattern::consumers_count(1));
    auto weights_pattern = pass::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // Can't fuse Multiply to Convolution if that Multiply is part of dequantization subgraph
        // since that breaks low precision transformations
        if (op::util::is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        auto weights_shape = weights.get_shape();
        std::shared_ptr<Node> mul_const = pattern_to_output.at(mul_const_pattern).get_node_shared_ptr();
        const auto& input = pattern_to_output.at(input_pattern);

        auto mul_const_shape = mul_const->output(0).get_partial_shape();
        auto input_shape = input.get_partial_shape();
        // Check if constant in multiply broadcasts input's shape.
        // If this is the case, we cannot perform the transformation as
        // 'bare' input's shape will not be aligned with weights.
        if (ov::op::util::check_for_broadcast(mul_const_shape, input_shape)) {
            return false;
        }

        if (shape_size(mul_const->get_shape()) > 1) {
            auto mul_const_shape = mul_const->get_shape();
            // extend mul_const_shape rank with unit dimensions
            if (weights_shape.size() > mul_const_shape.size())
                mul_const_shape.insert(mul_const_shape.begin(), weights_shape.size() - mul_const_shape.size(), 1);
            // Check if constant has following shape (1, C, 1, 1, ..)
            // We can't fuse constants like (1, C, H, W) due to backprop nature of this convolution
            // In backprop, weights pixels are applied to input differently than in fprop convolution
            for (size_t i = 0; i < mul_const_shape.size(); i++) {
                if (i == 1)
                    continue;
                if (mul_const_shape[i] != 1)
                    return false;
            }
            // Reshape mul_const from shape (1, C, 1, 1) to (C, 1, 1, 1) to match ConvolutionBackpropData weights format
            Shape new_shape{mul_const_shape[1], 1};
            new_shape.insert(new_shape.end(), mul_const_shape.size() - 2, 1);
            if (!ov::op::util::check_for_broadcast(weights_shape, new_shape)) {
                return false;
            }
            mul_const = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(element::u64, Shape{new_shape.size()}, new_shape),
                false);
        }

        auto weights_multiply = std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = ov::util::get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()}, {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::MultiplyGroupConvolutionBackpropDataFusion::MultiplyGroupConvolutionBackpropDataFusion() {
    MATCHER_SCOPE(MultiplyGroupConvolutionBackpropDataFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input_pattern, mul_const_pattern},
                                                                          pattern::consumers_count(1));
    auto weights_pattern = pass::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::GroupConvolutionBackpropData>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // Can't fuse Multiply to Convolution if that Multiply is part of dequantization subgraph
        // since that breaks low precision transformations
        if (op::util::is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        std::shared_ptr<Node> mul_const = pattern_to_output.at(mul_const_pattern).get_node_shared_ptr();
        const auto& input = pattern_to_output.at(input_pattern);

        auto mul_const_shape = mul_const->output(0).get_partial_shape();
        auto input_shape = input.get_partial_shape();
        // Check if constant in multiply broadcasts input's shape.
        // If this is the case, we cannot perform the transformation as
        // 'bare' input's shape will not be aligned with weights.
        if (ov::op::util::check_for_broadcast(mul_const_shape, input_shape)) {
            return false;
        }

        const auto& weights_shape = weights.get_shape();
        if (shape_size(mul_const->get_shape()) > 1) {
            auto mul_const_shape = mul_const->get_shape();
            // extend mul_const_shape rank with unit dimensions
            if (weights_shape.size() - mul_const_shape.size() > 1)
                mul_const_shape.insert(mul_const_shape.begin(), weights_shape.size() - mul_const_shape.size() - 1, 1);
            // We can't fuse constants like (1, C, H, W) due to backprop nature of this convolution
            // In backprop, weights pixels are applied to input differently than in fprop convolution
            for (size_t i = 0; i < mul_const_shape.size(); i++) {
                if (i == 1)
                    continue;
                if (mul_const_shape[i] != 1)
                    return false;
            }
            // Reshape mul_const from shape (1, C, 1, 1) to (G, C / G, 1, 1, 1) to match GroupConvolutionBackpropData
            // weights format
            auto G = mul_const_shape[1] > 1 ? weights_shape[0] : 1;
            auto C = mul_const_shape[1] / G;
            Shape new_shape{G, C, 1};
            new_shape.insert(new_shape.end(), mul_const_shape.size() - 2, 1);
            if (!ov::op::util::check_for_broadcast(weights_shape, new_shape)) {
                return false;
            }
            mul_const = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(element::u64, Shape{new_shape.size()}, new_shape),
                false);
        }

        auto weights_multiply = std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = ov::util::get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()}, {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}
