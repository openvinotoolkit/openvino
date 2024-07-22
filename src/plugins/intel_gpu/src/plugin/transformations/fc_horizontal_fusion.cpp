// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_horizontal_fusion.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/core/rt_info.hpp"
#include <openvino/opsets/opset1.hpp>
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/placeholder.hpp"

namespace ov {
namespace intel_gpu {

FullyConnectedHorizontalFusion::FullyConnectedHorizontalFusion() {
    using namespace ov::pass::pattern;

    auto is_target_pattern = [](const Output<Node>& output) {
        // Currently this pass targets only compressed FCs (QKV) on dynamic generative models
        // inputs: input, weight, bias, scale, [zp]
        // Bias/scale/zp are constant or none
        // if it is not constant, the only allowed cases are Constant => convert
        // All FCs have same # of valid inputs (e.g., if one of the fc has zp, all fcs have zp)

        auto is_constant = [](const std::shared_ptr<ov::Node> node) {
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(node))
                return true;
            if (std::dynamic_pointer_cast<ov::op::v0::Convert>(node) && std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0)))
                return true;
            if (std::dynamic_pointer_cast<ov::op::v1::Transpose>(node) && std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0)))
                return true;
            return false;
        };
        auto is_placeholder = [](const std::shared_ptr<ov::Node> node) {
            return std::dynamic_pointer_cast<op::Placeholder>(node);
        };
        // Three FCs connected to the same input
        const int num_fcs_to_fuse = 3;
        const auto& fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(output.get_node_shared_ptr());
        const auto& input = fc->get_input_node_shared_ptr(0);
        if (!fc->get_input_partial_shape(0).is_dynamic())
            return false;
        if (input->get_users().size() < num_fcs_to_fuse)
            return false;
        size_t user_fc_count = 0;
        int32_t nodes_with_bias = 0;
        int32_t nodes_with_zp = 0;
        for (const auto& u : input->get_users()) {
            const auto& fc_user = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(u);
            if (!fc_user)
                continue;
            auto num_inputs = fc_user->inputs().size();
            if (num_inputs >= 5)
                nodes_with_zp++;
            for (size_t i = 2; i < num_inputs; ++i) {
                const auto& fc_input = fc_user->get_input_node_shared_ptr(i);
                if (!is_constant(fc_input) && !is_placeholder(fc_input))
                    return false;
                if (i == 2 && !is_placeholder(fc_input)) {
                    nodes_with_bias++;
                }
            }
            user_fc_count++;
        }
        return (user_fc_count == num_fcs_to_fuse) && (nodes_with_bias == num_fcs_to_fuse || nodes_with_bias == 0) &&
               (nodes_with_zp == num_fcs_to_fuse || nodes_with_zp == 0);
    };

    auto target_fc = wrap_type<op::FullyConnectedCompressed>(is_target_pattern);

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto m_fc = pattern_map.at(target_fc).get_node_shared_ptr();
        auto input_node = m_fc->get_input_node_shared_ptr(0);
        std::vector<std::shared_ptr<op::FullyConnectedCompressed>> fc_nodes;
        ov::NodeVector weight_nodes;
        ov::NodeVector scale_nodes;
        ov::NodeVector bias_nodes;
        ov::NodeVector zp_nodes;
        for (auto user : input_node->get_users()) {
            auto fc_user = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(user);
            if (fc_user) {
                OPENVINO_ASSERT(fc_user->inputs().size() >= 4, "Compressed FC should have at least 4 inputs");
                fc_nodes.push_back(fc_user);
                weight_nodes.push_back(fc_user->get_input_node_shared_ptr(1));
                if (!std::dynamic_pointer_cast<op::Placeholder>(fc_user->get_input_node_shared_ptr(2)))
                    bias_nodes.push_back(fc_user->get_input_node_shared_ptr(2));
                scale_nodes.push_back(fc_user->get_input_node_shared_ptr(3));
                if (fc_user->inputs().size() > 4)
                    zp_nodes.push_back(fc_user->get_input_node_shared_ptr(4));
            }
        }
        auto weight_dtype = fc_nodes[0]->get_input_element_type(1);
        auto k_size = fc_nodes[0]->get_input_shape(1)[fc_nodes[0]->get_input_shape(1).size() - 1];
        std::vector<int64_t> orig_n_sizes;
        // merge weights, scale, zp
        for (auto fc : fc_nodes) {
            if (k_size != fc->get_input_shape(1)[fc->get_input_shape(1).size() - 1])
                return false;
            if (weight_dtype != fc->get_input_element_type(1))
                return false;
            orig_n_sizes.push_back(fc->get_input_shape(1)[fc->get_input_shape(1).size() - 2]);
        }
        auto weight_nodes_as_output_vector = ov::OutputVector{weight_nodes[0], weight_nodes[1], weight_nodes[2]};
        auto fused_weight = std::make_shared<ov::op::v0::Concat>(weight_nodes_as_output_vector, 0);
        fused_weight->set_friendly_name(weight_nodes[0]->get_friendly_name() + "_fused");
        ov::copy_runtime_info({weight_nodes[0], weight_nodes[1], weight_nodes[2]}, fused_weight);

        auto scale_nodes_as_output_vector = ov::OutputVector{scale_nodes[0], scale_nodes[1], scale_nodes[2]};
        auto fused_scale = std::make_shared<ov::op::v0::Concat>(scale_nodes_as_output_vector, 0);
        fused_scale->set_friendly_name(scale_nodes[0]->get_friendly_name() + "_fused");
        ov::copy_runtime_info({scale_nodes[0], scale_nodes[1], scale_nodes[2]}, fused_scale);

        std::shared_ptr<ov::Node> fused_bias;
        if (bias_nodes.size() == 3) {
            auto bias_nodes_as_output_vector = ov::OutputVector{bias_nodes[0], bias_nodes[1], bias_nodes[2]};
            fused_bias = std::make_shared<ov::op::v0::Concat>(bias_nodes_as_output_vector, 0);
            fused_bias->set_friendly_name(bias_nodes[0]->get_friendly_name() + "_fused");
            ov::copy_runtime_info({bias_nodes[0], bias_nodes[1], bias_nodes[2]}, fused_bias);
        } else {
            fused_bias = std::make_shared<op::Placeholder>();
        }

        std::shared_ptr<ov::Node> fused_zps;
        if (zp_nodes.size() > 0) {
            // scalar zp
            auto zp_shape = zp_nodes[0]->get_output_shape(0);
            bool is_scalar = (ov::shape_size(zp_nodes[0]->get_output_shape(0)) == 1);
            int32_t scalar_zp_val = 0;
            if (is_scalar) {
                if (auto zp_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(zp_nodes[0])) {
                    scalar_zp_val = zp_const->cast_vector<int32_t>()[0];
                } else if (auto zp_convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(zp_nodes[0])) {
                    auto zp_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(zp_convert->get_input_node_shared_ptr(0));
                    scalar_zp_val = zp_const->cast_vector<int32_t>()[0];
                }
                fused_zps = zp_nodes[0];
            }
            if (is_scalar) {
                for (size_t i = 1; i < zp_nodes.size(); ++i) {
                    bool current_is_scalar = (ov::shape_size(zp_nodes[i]->get_output_shape(0)) == 1);
                    if (!current_is_scalar)
                        return false;
                    // validate all zp values are same
                    int32_t cur_zp_val = 0;
                    if (auto zp_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(zp_nodes[i])) {
                        cur_zp_val = zp_const->cast_vector<int32_t>()[0];
                    } else if (auto zp_convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(zp_nodes[i])) {
                        auto zp_const =
                            std::dynamic_pointer_cast<ov::op::v0::Constant>(zp_convert->get_input_node_shared_ptr(0));
                        cur_zp_val = zp_const->cast_vector<int32_t>()[0];
                    } else {
                        OPENVINO_ASSERT("Unsupported zp input node for FC horizontal fusion");
                    }
                    if (cur_zp_val != scalar_zp_val)
                        return false;
                }
            } else {
                auto zp_nodes_as_output_vector = ov::OutputVector{zp_nodes[0], zp_nodes[1], zp_nodes[2]};
                fused_zps = std::make_shared<ov::op::v0::Concat>(zp_nodes_as_output_vector, 0);
                fused_zps->set_friendly_name(zp_nodes[0]->get_friendly_name() + "_fused");
            }
        }
        // Create new fc with merged weights, bias, scale, zp
        std::shared_ptr<ov::Node> new_fc;
        if (fused_zps)
            new_fc = std::make_shared<op::FullyConnectedCompressed>(input_node, fused_weight, fused_bias, fused_scale, fused_zps);
        else
            new_fc = std::make_shared<op::FullyConnectedCompressed>(input_node, fused_weight, fused_bias, fused_scale);

        auto new_fc_name = fc_nodes[0]->get_friendly_name() + "_fused";
        new_fc->set_friendly_name(new_fc_name);
        copy_runtime_info({fc_nodes[0], fc_nodes[1], fc_nodes[2]}, new_fc);

        // Split output and connect to the orig users
        auto split_name = fc_nodes[0]->get_friendly_name() + "_split";
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {new_fc->get_output_partial_shape(0).size() - 1});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
        auto output_split = std::make_shared<ov::op::v1::VariadicSplit>(new_fc, axis_const, split_const);
        copy_runtime_info({fc_nodes[0], fc_nodes[1], fc_nodes[2]}, output_split);
        output_split->set_friendly_name(split_name);
        for (size_t i = 0; i < fc_nodes.size(); ++i) {
            auto org_fc = fc_nodes[i];
            for (auto u : org_fc->get_users()) {
                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                    if (u->get_input_node_shared_ptr(idx) == org_fc) {
                        u->input(idx).replace_source_output(output_split->output(i));
                    }
                }
            }
            org_fc->clear_control_dependencies();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(target_fc, "FullyConnectedHorizontalFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
