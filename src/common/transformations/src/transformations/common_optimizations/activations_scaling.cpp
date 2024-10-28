// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::ActivationsScaling::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ActivationsScaling);

    if (m_scale_factor < 1.f)
        return false;

    std::cout << "scale_factor: " << m_scale_factor << std::endl;

    std::unordered_set<std::string> scaled_down_nodes;
    std::unordered_set<std::string> normal_nodes;
    std::unordered_set<std::string> constant_nodes;

    ov::Shape scale_const_shape = {1};
    std::vector<float> scale_value = {m_scale_factor};
    std::vector<float> inverse_scale_value = {(1.f / m_scale_factor)};
    std::shared_ptr<ov::Node> scale_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_value);
    std::shared_ptr<ov::Node> scale_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_value);
    std::shared_ptr<ov::Node> inverse_scale_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, inverse_scale_value);
    std::shared_ptr<ov::Node> inverse_scale_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, inverse_scale_value);

    for (auto& node : f->get_ordered_ops()) {
        auto parameter_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node);
        if (parameter_node &&
            (parameter_node->get_element_type() == ov::element::f16 ||
             parameter_node->get_element_type() == ov::element::f32)) {
            std::shared_ptr<ov::Node> inverse_scale_const = (parameter_node->get_element_type() == ov::element::f16) ?
                                                             inverse_scale_const_f16 : inverse_scale_const_f32;
            auto scale_down = std::make_shared<ov::op::v1::Multiply>(parameter_node->output(0),
                                                                     inverse_scale_const->output(0));
            ov::replace_node(parameter_node, scale_down);
            scaled_down_nodes.insert(node->get_friendly_name());
            scaled_down_nodes.insert(scale_down->get_friendly_name());
            continue;
        }

        auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
        if (const_node) {
            constant_nodes.insert(node->get_friendly_name());
            continue;
        }

        auto group_norm_node = std::dynamic_pointer_cast<ov::op::v12::GroupNormalization>(node);
        auto mvn_node = std::dynamic_pointer_cast<ov::op::v6::MVN>(node);
        if (group_norm_node || mvn_node) {
            normal_nodes.insert(node->get_friendly_name());
            continue;
        }

        size_t num_scaled_down_inputs = 0;
        size_t num_const_inputs = 0;
        size_t num_normal_inputs = 0;
        for (auto& dep: node->inputs()) {
            auto dep_name = dep.get_source_output().get_node_shared_ptr()->get_friendly_name();

            if (scaled_down_nodes.find(dep_name) != scaled_down_nodes.end()) {
                num_scaled_down_inputs += 1;
                continue;
            }
            if (constant_nodes.find(dep_name) != constant_nodes.end()) {
                num_const_inputs += 1;
                continue;
            }
            if (normal_nodes.find(dep_name) != normal_nodes.end()) {
                num_normal_inputs += 1;
                continue;
            }
        }

        if (node->get_input_size() > 0) {
            if (num_const_inputs == node->get_input_size()) {
                constant_nodes.insert(node->get_friendly_name());
                continue;
            }
            if ((num_const_inputs + num_normal_inputs) == node->get_input_size()) {
                normal_nodes.insert(node->get_friendly_name());
                continue;
            }
        }

        if (num_scaled_down_inputs == 0) {
            continue;
        }

        auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(node);
        if (result && num_scaled_down_inputs == 1) {
            auto dep = node->input(0);
            std::shared_ptr<ov::Node> scale_const = (dep.get_element_type() == ov::element::f16) ?
                                                    scale_const_f16 : scale_const_f32;
            auto scale_up = std::make_shared<ov::op::v1::Multiply>(dep.get_source_output(), scale_const->output(0));
            dep.replace_source_output(scale_up->output(0));
        }

        //    input0         input1            input0         input1
        // (scaled_down)  (non-scaled)      (scaled_down)  (non-scaled)
        //          \         /                      \         /
        //           \       /         ==>            \    scale_down
        //            \     /                          \     /
        //              add                              add
        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(node);
        if (add && num_scaled_down_inputs == 1) {
            for (auto& dep: node->inputs()) {
                if (scaled_down_nodes.find(dep.get_source_output().get_node_shared_ptr()->get_friendly_name()) == scaled_down_nodes.end()) {
                    std::shared_ptr<ov::Node> inverse_scale_const = (dep.get_element_type() == ov::element::f16) ?
                                                                    inverse_scale_const_f16 : inverse_scale_const_f32;
                    auto scale_down = std::make_shared<ov::op::v1::Multiply>(dep.get_source_output(),
                                                                             inverse_scale_const->output(0));
                    dep.replace_source_output(scale_down->output(0));
                }
            }
        }

        auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(node);
        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        if ((multiply || matmul) && num_scaled_down_inputs == 2) {
            auto dep = node->input(1);
            std::shared_ptr<ov::Node> scale_const = (dep.get_element_type() == ov::element::f16) ?
                                                    scale_const_f16 : scale_const_f32;
            auto scale_up = std::make_shared<ov::op::v1::Multiply>(dep.get_source_output(), scale_const->output(0));
            dep.replace_source_output(scale_up->output(0));
        }

        auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node);
        if (sdpa) {
            for (size_t i = 0; i < 2; i++) {
                if (scaled_down_nodes.find(node->input(i).get_source_output().get_node_shared_ptr()->get_friendly_name()) != scaled_down_nodes.end()) {
                    std::shared_ptr<ov::Node> scale_const = (node->get_input_element_type(i) == ov::element::f16) ? scale_const_f16 : scale_const_f32;
                    auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(node->input(i).get_source_output().get_node_shared_ptr());
                    if (transpose) {
                        auto scale_up = std::make_shared<ov::op::v1::Multiply>(transpose->get_input_source_output(0),
                                                                               scale_const->output(0));
                        transpose->input(0).replace_source_output(scale_up->output(0));
                    } else {
                        auto scale_up = std::make_shared<ov::op::v1::Multiply>(node->get_input_source_output(i),
                                                                               scale_const->output(0));
                        node->input(i).replace_source_output(scale_up->output(0));
                    }
                }
            }

            if (scaled_down_nodes.find(node->input(2).get_source_output().get_node_shared_ptr()->get_friendly_name()) != scaled_down_nodes.end()) {
                scaled_down_nodes.insert(node->get_friendly_name());
            }
            continue;
        }

        // input(scaled_down) -- activation
        // ==>
        // input(scaled_down) -- convert(precision_up) -- multiply(scale_up) -- activation -- multiply(scale_down) -- convert(precision_down)
        auto sin = std::dynamic_pointer_cast<ov::op::v0::Sin>(node);
        auto cos = std::dynamic_pointer_cast<ov::op::v0::Cos>(node);
        auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(node);
        auto power = std::dynamic_pointer_cast<ov::op::v1::Power>(node);
        auto sqrt = std::dynamic_pointer_cast<ov::op::v0::Sqrt>(node);
        auto gelu = std::dynamic_pointer_cast<ov::op::v7::Gelu>(node);
        auto softmax = std::dynamic_pointer_cast<ov::op::v8::Softmax>(node);
        if ((sin || cos || swish || power || sqrt || gelu || softmax) && num_scaled_down_inputs == 1) {
            auto input_prec = node->get_input_element_type(0);
            auto output_prec = node->get_output_element_type(0);

            ov::Output<ov::Node> input_src;
            if (input_prec == ov::element::f16) {
                auto precision_up = std::make_shared<ov::op::v0::Convert>(node->get_input_source_output(0), ov::element::f32);
                input_src = precision_up->output(0);
            } else {
                input_src = node->get_input_source_output(0);
            }
            auto scale_up = std::make_shared<ov::op::v1::Multiply>(input_src,
                                                                   scale_const_f32->output(0));
            node->input(0).replace_source_output(scale_up->output(0));
            node->revalidate_and_infer_types();

            auto scale_down = std::make_shared<ov::op::v1::Multiply>(node->output(0),
                                                                     inverse_scale_const_f32->output(0));
            ov::replace_node(node, scale_down);
            scaled_down_nodes.insert(scale_down->get_friendly_name());

            if (output_prec == ov::element::f16) {
                auto precision_down = std::make_shared<ov::op::v0::Convert>(scale_down->output(0), ov::element::f16);
                ov::replace_node(scale_down, precision_down);
                scaled_down_nodes.insert(precision_down->get_friendly_name());
            }
        }

        if (num_scaled_down_inputs > 0)
            scaled_down_nodes.insert(node->get_friendly_name());
    }

    return true;
}
