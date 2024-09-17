// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::AddFakeQuantizeFusion::AddFakeQuantizeFusion() {
    MATCHER_SCOPE(AddFakeQuantizeFusion);
    auto input_pattern = pass::pattern::any_input();
    auto const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Add>({input_pattern, const_pattern}, pattern::consumers_count(1));
    auto fq_pattern = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>({add_pattern,
                                                                              pass::pattern::any_input(),
                                                                              pass::pattern::any_input(),
                                                                              pass::pattern::any_input(),
                                                                              pass::pattern::any_input()});
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input = pattern_value_map.at(input_pattern);
        const auto& type = input.get_element_type();
        if (type.bitwidth() < element::f32.bitwidth())
            return false;
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(pattern_value_map.at(fq_pattern).get_node_shared_ptr());
        if (!fq)
            return false;
        const auto& add_node = pattern_value_map.at(add_pattern).get_node_shared_ptr();
        auto add_const =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_value_map.at(const_pattern).get_node_shared_ptr());
        if (!add_const)
            return false;

        auto const_shape = add_const->get_shape();
        if (!ov::op::util::check_for_broadcast(input.get_partial_shape(), const_shape)) {
            // We can't eliminate Add if Constant input broadcasts another input shape because
            // when we reconnect input from Add to FQ won't broadcast given input, so it will result
            // in shape collision.
            return false;
        }

        std::shared_ptr<Node> new_const = add_const;
        size_t const_shape_size = shape_size(const_shape);
        bool is_single_value = const_shape_size == 1;

        if (!is_single_value) {
            float v;
            is_single_value = op::util::get_single_value(add_const, v);
            if (is_single_value) {
                new_const = std::make_shared<ov::op::v0::Constant>(add_const->get_element_type(), Shape{1}, v);
            }
        }

        if (!is_single_value) {
            const auto& fq_input_shape = fq->get_input_partial_shape(0);
            if (fq_input_shape.rank().is_dynamic())
                return false;

            const auto diff = fq_input_shape.size() - const_shape.size();
            if (diff > 0) {
                // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
                const_shape.insert(const_shape.begin(), diff, 1);
                new_const = std::make_shared<ov::op::v1::Reshape>(
                    new_const,
                    ov::op::v0::Constant::create(element::u64, Shape{const_shape.size()}, const_shape),
                    false);
            }

            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(const_shape[0] > 1 && const_shape[0] == const_shape_size) &&
                !(const_shape.size() > 1 && const_shape[1] == const_shape_size)) {
                return false;
            }

            // Convolution+Add or MatMul+Add can be fused later
            // so don't fuse Add+FQ in that situation
            const auto& add_inputs = add_node->input_values();
            bool add_parent_is_conv_or_mm =
                std::any_of(add_inputs.begin(), add_inputs.end(), [](const Output<Node>& node) -> bool {
                    auto node_ptr = node.get_node();
                    return is_type<ov::op::v1::Convolution>(node_ptr) ||
                           is_type<ov::op::v1::GroupConvolution>(node_ptr) ||
                           is_type<ov::op::v1::ConvolutionBackpropData>(node_ptr) ||
                           is_type<ov::op::v1::GroupConvolutionBackpropData>(node_ptr) ||
                           is_type<ov::op::v0::MatMul>(node_ptr);
                });
            if (add_parent_is_conv_or_mm)
                return false;
            auto fq_users = fq->get_users();
            // Concat LPT transformation supports per tensor quantization only
            bool fq_user_is_concat =
                std::any_of(fq_users.begin(), fq_users.end(), [](const std::shared_ptr<Node> node_ptr) -> bool {
                    return is_type<ov::op::v0::Concat>(node_ptr);
                });
            if (fq_user_is_concat)
                return false;
        }

        auto input_low_sub = std::make_shared<ov::op::v1::Subtract>(fq->input_value(1), new_const);
        std::shared_ptr<Node> new_input_low = ov::util::get_constant_from_source(input_low_sub);
        if (!new_input_low)
            new_input_low = input_low_sub;
        auto input_high_sub = std::make_shared<ov::op::v1::Subtract>(fq->input_value(2), new_const);
        std::shared_ptr<Node> new_input_high = ov::util::get_constant_from_source(input_high_sub);
        if (!new_input_high)
            new_input_high = input_high_sub;
        auto new_fq =
            fq->clone_with_new_inputs({input, new_input_low, new_input_high, fq->input_value(3), fq->input_value(4)});
        if (transformation_callback(new_fq))
            return false;
        register_new_node(new_fq);
        new_fq->set_friendly_name(fq->get_friendly_name());
        copy_runtime_info({add_node, fq}, {new_input_low, new_input_high, new_fq});
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
