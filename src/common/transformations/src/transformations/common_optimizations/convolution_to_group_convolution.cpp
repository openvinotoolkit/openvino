// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/convolution_to_group_convolution_fusion.hpp"

static bool compare_convolutions(const ov::op::v1::Convolution* conv1, ov::Node* node) {
    const auto conv2 = ov::as_type<ov::op::v1::Convolution>(node);
    if (!conv2)
        return false;
    return conv1->get_strides() == conv2->get_strides() && conv1->get_pads_begin() == conv2->get_pads_begin() &&
           conv1->get_pads_end() == conv2->get_pads_end() && conv1->get_dilations() == conv2->get_dilations() &&
           conv1->get_auto_pad() == conv2->get_auto_pad();
}

static int64_t get_split_axis(const std::shared_ptr<ov::Node>& split) {
    const auto axis = ov::as_type<ov::op::v0::Constant>(split->get_input_node_ptr(1));
    if (!axis)
        return -1;
    auto axis_value = axis->cast_vector<int64_t>()[0];
    if (axis_value < 0) {
        const auto& input_rank = split->get_input_partial_shape(0).rank();
        if (input_rank.is_dynamic())
            return -1;
        axis_value += input_rank.get_length();
    }

    return axis_value;
}

static std::shared_ptr<ov::op::v0::Concat> create_new_weights(ov::pass::NodeRegistry& node_registry,
                                                              const std::shared_ptr<ov::Node>& concat) {
    const auto concat_input = concat->get_input_node_ptr(0);
    if (concat_input->get_input_partial_shape(1).is_dynamic())
        return nullptr;

    // unsqueeze weights shape from (O, I, X, Y) to (1, O, I, X, Y)
    const auto& weights_shape = concat_input->get_input_shape(1);
    ov::Shape new_shape = weights_shape;
    new_shape.insert(new_shape.begin(), 1);

    const size_t num_inputs = concat->get_input_size();
    ov::OutputVector weights_to_concat;
    weights_to_concat.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
        const auto conv = concat->get_input_node_shared_ptr(i);
        const auto weights = conv->get_input_node_shared_ptr(1);
        const auto& shape = weights->get_output_partial_shape(0);
        if (shape.is_dynamic() || weights->get_output_shape(0) != weights_shape)
            return nullptr;
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(weights)) {
            weights_to_concat.push_back(node_registry.make<ov::op::v0::Constant>(*constant, new_shape));
        } else {
            weights_to_concat.push_back(node_registry.make<ov::op::v0::Unsqueeze>(
                weights,
                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0})));
        }
        weights_to_concat.back().get_node()->set_friendly_name(weights->get_friendly_name());
    }

    return node_registry.make<ov::op::v0::Concat>(weights_to_concat, 0);
}

ov::pass::ConvolutionToGroupConvolutionFusion::ConvolutionToGroupConvolutionFusion() {
    MATCHER_SCOPE(ConvolutionToGroupConvolutionFusion);

    auto has_conv_inputs = [](const Output<Node>& node) -> bool {
        const auto concat = node.get_node();
        size_t num_inputs = concat->get_input_size();
        if (num_inputs == 0)
            return false;

        const auto first_conv = as_type<ov::op::v1::Convolution>(concat->get_input_node_ptr(0));
        if (!first_conv)
            return false;

        const auto split = first_conv->get_input_node_ptr(0);
        if (!is_type<ov::op::v1::Split>(split) && !is_type<ov::op::v1::VariadicSplit>(split))
            return false;

        // go through Concat inputs and check
        // - if all of them are Convolutions
        // - if those Convolutions have the same Split input
        for (size_t i = 1; i < concat->get_input_size(); i++) {
            const auto conv = concat->get_input_node_ptr(i);
            if (conv->get_input_node_ptr(0) != split)
                return false;
            if (!compare_convolutions(first_conv, conv))
                return false;
        }
        return true;
    };
    auto concat_label = pattern::wrap_type<ov::op::v0::Concat>(has_conv_inputs);

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& concat = pattern_value_map.at(concat_label).get_node_shared_ptr();

        const auto first_conv = as_type_ptr<ov::op::v1::Convolution>(concat->get_input_node_shared_ptr(0));
        const auto split = first_conv->get_input_node_shared_ptr(0);
        const bool is_split = is_type<ov::op::v1::Split>(split);
        const bool is_variadic_split = is_type<ov::op::v1::VariadicSplit>(split);
        if (!is_split && !is_variadic_split)
            return false;

        if (get_split_axis(split) != 1)
            return false;

        if (is_variadic_split) {
            // split_lengths in VariadicSplit must have the same values
            if (auto split_lengths = as_type<ov::op::v0::Constant>(split->get_input_node_ptr(1))) {
                const auto split_lengths_values = split_lengths->cast_vector<int>();
                const auto first_length = split_lengths_values[0];
                if (!std::all_of(split_lengths_values.begin() + 1,
                                 split_lengths_values.end(),
                                 [first_length](int split_length) {
                                     return split_length == first_length;
                                 }))
                    return false;
            } else {
                return false;
            }
        }

        NodeRegistry node_registry;
        const auto weights = create_new_weights(node_registry, concat);
        if (!weights)
            return false;

        const auto conv = node_registry.make<ov::op::v1::GroupConvolution>(split->get_input_node_shared_ptr(0),
                                                                           weights,
                                                                           first_conv->get_strides(),
                                                                           first_conv->get_pads_begin(),
                                                                           first_conv->get_pads_end(),
                                                                           first_conv->get_dilations(),
                                                                           first_conv->get_auto_pad());
        conv->set_friendly_name(concat->get_friendly_name());
        register_new_node(conv);

        const size_t concat_num_inputs = concat->get_input_size();
        NodeVector from;
        from.reserve(concat_num_inputs + 2);
        from.push_back(split);
        from.push_back(first_conv);
        for (size_t i = 1; i < concat_num_inputs; i++) {
            from.push_back(concat->get_input_node_shared_ptr(i));
        }
        from.push_back(concat);

        copy_runtime_info(from, node_registry.get());
        replace_node(concat, conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_label, matcher_name);
    this->register_matcher(m, callback);
}
