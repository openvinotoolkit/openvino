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
#include "intel_gpu/runtime/debug_configuration.hpp"

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
        const int min_num_fcs_to_fuse = 3;
        const int max_num_fcs_to_fuse = 3;
        const auto& fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(output.get_node_shared_ptr());
        const auto& input = fc->get_input_node_shared_ptr(0);
        if (!fc->get_input_partial_shape(0).is_dynamic())
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
        return (user_fc_count >= min_num_fcs_to_fuse) && (user_fc_count <= max_num_fcs_to_fuse) &&
               (nodes_with_bias == static_cast<int32_t>(user_fc_count) || nodes_with_bias == 0) &&
               (nodes_with_zp == static_cast<int32_t>(user_fc_count) || nodes_with_zp == 0);
    };

    auto target_fc = wrap_type<op::FullyConnectedCompressed>(is_target_pattern);

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto m_fc = pattern_map.at(target_fc).get_node_shared_ptr();
        auto input_node = m_fc->get_input_node_shared_ptr(0);
        std::vector<std::shared_ptr<op::FullyConnectedCompressed>> fc_nodes;
        ov::NodeVector fc_nodes_vec;
        ov::NodeVector weight_nodes;
        ov::NodeVector scale_nodes;
        ov::NodeVector bias_nodes;
        ov::NodeVector zp_nodes;
        int32_t bias_rank = -1;
        for (auto user : input_node->get_users()) {
            auto fc_user = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(user);
            if (fc_user) {
                OPENVINO_ASSERT(fc_user->inputs().size() >= 4, "Compressed FC should have at least 4 inputs");
                fc_nodes.push_back(fc_user);
                fc_nodes_vec.push_back(fc_user);
                weight_nodes.push_back(fc_user->get_input_node_shared_ptr(1));
                if (!std::dynamic_pointer_cast<op::Placeholder>(fc_user->get_input_node_shared_ptr(2))) {
                    if (bias_rank == -1)
                        bias_rank = static_cast<int32_t>(fc_user->get_input_partial_shape(2).size());
                    if (bias_rank != static_cast<int32_t>(fc_user->get_input_partial_shape(2).size()))
                        return false;
                    bias_nodes.push_back(fc_user->get_input_node_shared_ptr(2));
                }
                scale_nodes.push_back(fc_user->get_input_node_shared_ptr(3));
                if (fc_user->inputs().size() > 4)
                    zp_nodes.push_back(fc_user->get_input_node_shared_ptr(4));
            }
        }
        // fc weight is already transposed to [N, K]
        const size_t weight_idx = 1;
        if (fc_nodes[0]->get_input_shape(weight_idx).size() != 2)
            return false;
        const size_t n_axis = 0;
        const size_t k_axis = 1;
        auto weight_dtype = fc_nodes[0]->get_input_element_type(weight_idx);
        auto k_size = fc_nodes[0]->get_input_shape(weight_idx)[k_axis];
        std::vector<int64_t> orig_n_sizes;
        // merge weights, scale, zp
        for (auto fc : fc_nodes) {
            if (k_size != fc->get_input_shape(weight_idx)[k_axis])
                return false;
            if (weight_dtype != fc->get_input_element_type(weight_idx))
                return false;
            orig_n_sizes.push_back(fc->get_input_shape(weight_idx)[n_axis]);
        }
        ov::OutputVector weight_nodes_as_output_vector;
        for (size_t i = 0; i < weight_nodes.size(); ++i) {
            weight_nodes_as_output_vector.push_back(weight_nodes[i]);
        }
        auto fused_weight = std::make_shared<ov::op::v0::Concat>(weight_nodes_as_output_vector, 0);
        fused_weight->set_friendly_name(weight_nodes[0]->get_friendly_name() + "_fused_weight");
        ov::copy_runtime_info(weight_nodes, fused_weight);

        ov::OutputVector scales_as_output_vector;
        for (size_t i = 0; i < scale_nodes.size(); ++i) {
            scales_as_output_vector.push_back(scale_nodes[i]);
        }

        auto fused_scale = std::make_shared<ov::op::v0::Concat>(scales_as_output_vector, 0);
        fused_scale->set_friendly_name(scale_nodes[0]->get_friendly_name() + "_fused_scale");
        ov::copy_runtime_info(scale_nodes, fused_scale);
        // check if the FCs do not have bias inputs, but all of the fc has a bias add user, set them as bias inputs
        // Currently horizontal fusing is applied only when fusing is applied for N dim
        // Also, fuse biases for the last dimension too, if
        // - Biases are constant
        // - Rank of the bias shapes are same
        // - all other dims except last dim is 1 (e.g., [1, 1, N])
        size_t n_bias_users = 0;
        if (bias_nodes.empty()) {
            for (auto fc : fc_nodes) {
                if (fc->get_users().size() == 1 &&
                    fc->get_users()[0]->get_type_info() == ov::opset1::Add::get_type_info_static() &&
                    ov::is_type<ov::op::v0::Constant>(fc->get_users()[0]->inputs()[1].get_source_output().get_node())) {
                    auto bias_input1_shape = fc->get_users()[0]->get_input_partial_shape(1).get_shape();
                    if (bias_rank == -1)
                        bias_rank = static_cast<int32_t>(bias_input1_shape.size());
                    if (bias_rank != static_cast<int32_t>(bias_input1_shape.size()))
                        break;
                    size_t ndim_size = bias_input1_shape.back();
                    // allow only [1, 1, N] shape bias
                    if (std::accumulate(bias_input1_shape.begin(),
                                        bias_input1_shape.end(),
                                        static_cast<size_t>(1),
                                        std::multiplies<size_t>()) != ndim_size)
                        break;
                    n_bias_users++;
                }
            }

            if (n_bias_users == fc_nodes.size()) {
                for (size_t i = 0; i < fc_nodes.size(); ++i) {
                    auto orig_fc = fc_nodes[i];
                    auto bias_node = orig_fc->get_users()[0];
                    auto bias_const_ptr = orig_fc->get_users()[0]->get_input_node_shared_ptr(1);
                    bias_nodes.push_back(bias_const_ptr);
                }
                for (size_t i = 0; i < fc_nodes.size(); ++i) {
                    auto orig_fc = fc_nodes[i];
                    auto bias_node = orig_fc->get_users()[0];
                    GPU_DEBUG_TRACE_DETAIL << "Set Add op user " << bias_node->get_friendly_name() << " as the FC "
                                           << orig_fc->get_friendly_name() << "'s bias input" << std::endl;
                    auto bias_const = orig_fc->get_users()[0]->input_value(1);
                    auto orig_users_of_bias_user = bias_node->get_users();
                    ov::OutputVector fc_inputs = orig_fc->input_values();
                    fc_inputs[2] = bias_const;
                    auto new_fc = orig_fc->clone_with_new_inputs(fc_inputs);
                    new_fc->set_friendly_name(orig_fc->get_friendly_name() + "_with_bias");
                    ov::copy_runtime_info(orig_fc, new_fc);
                    for (auto u : orig_users_of_bias_user) {
                        for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                            if (u->get_input_node_shared_ptr(idx) == bias_node) {
                                u->input(idx).replace_source_output(new_fc->output(0));
                            }
                        }
                    }
                    fc_nodes[i] = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(new_fc);
                    bias_node->clear_control_dependencies();
                    orig_fc->clear_control_dependencies();
                }
            }
        }

        std::shared_ptr<ov::Node> fused_bias;
        if (bias_nodes.size() == fc_nodes.size()) {
            ov::OutputVector bias_nodes_as_output_vector;
            for (size_t i = 0; i < bias_nodes.size(); ++i) {
                bias_nodes_as_output_vector.push_back(bias_nodes[i]);
            }
            fused_bias = std::make_shared<ov::op::v0::Concat>(bias_nodes_as_output_vector, bias_rank - 1);
            fused_bias->set_friendly_name(bias_nodes[0]->get_friendly_name() + "_fused_bias");
            ov::copy_runtime_info(bias_nodes, fused_bias);
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
                ov::OutputVector zp_nodes_as_output_vector;
                for (size_t i = 0; i < zp_nodes.size(); ++i) {
                    zp_nodes_as_output_vector.push_back(zp_nodes[i]);
                }
                fused_zps = std::make_shared<ov::op::v0::Concat>(zp_nodes_as_output_vector, 0);
                fused_zps->set_friendly_name(zp_nodes[0]->get_friendly_name() + "_fused_zps");
            }
        }
        // Create new fc with merged weights, bias, scale, zp
        std::shared_ptr<ov::Node> new_fc;
        if (fused_zps)
            new_fc = std::make_shared<op::FullyConnectedCompressed>(input_node,
                                                                    fused_weight,
                                                                    fused_bias,
                                                                    fused_scale,
                                                                    fused_zps,
                                                                    fc_nodes[0]->get_output_type());
        else
            new_fc = std::make_shared<op::FullyConnectedCompressed>(input_node,
                                                                    fused_weight,
                                                                    fused_bias,
                                                                    fused_scale,
                                                                    fc_nodes[0]->get_output_type());

        auto new_fc_name = fc_nodes[0]->get_friendly_name() + "_fused_" + std::to_string(fc_nodes.size()) + "FCs";
        new_fc->set_friendly_name(new_fc_name);
        copy_runtime_info(fc_nodes_vec, new_fc);

        // Split output and connect to the orig users
        auto split_name = fc_nodes[0]->get_friendly_name() + "_split";
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {new_fc->get_output_partial_shape(0).size() - 1});
        auto split_size = fc_nodes.size();
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{split_size}, orig_n_sizes);
        auto output_split = std::make_shared<ov::op::v1::VariadicSplit>(new_fc, axis_const, split_const);
        copy_runtime_info(fc_nodes_vec, output_split);
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
        GPU_DEBUG_TRACE_DETAIL << "Created a new fused FC " << new_fc_name << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(target_fc, "FullyConnectedHorizontalFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
