// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_horizontal_fusion.hpp"

#include <string_view>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

FullyConnectedHorizontalFusion::FullyConnectedHorizontalFusion(bool fuse_mlp_swiglu) {
    using namespace ov::pass::pattern;

    // Three FCs connected to the same input
    size_t min_num_fcs_to_fuse = 3;
    // Note:
    // For cldnn, two fcs in mlp will be fused at horizontal fc fusion, and then swiglu will be fused at prepare_primitive_fusion
    // i.e., eltwise((fc + swish), fc) => fused_fc + swiglu => fused_fc_swilgu
    // Onednn gemms are to be handled in a different way (TBD)
    if (fuse_mlp_swiglu) {
        min_num_fcs_to_fuse = 2;
    }
    auto is_target_pattern = [min_num_fcs_to_fuse](const Output<Node>& output) {
        const int max_num_fcs_to_fuse = 3;
        // Currently this pass targets only compressed FCs (QKV) on dynamic generative models
        // inputs: input, weight, bias, scale, [zp]
        // Bias/scale/zp are constant or none
        // if it is not constant, the only allowed cases are Constant => convert
        // All FCs have same # of valid inputs (e.g., if one of the fc has zp, all fcs have zp)
        auto is_constant = [](const std::shared_ptr<ov::Node> node) {
            if (ov::as_type_ptr<ov::op::v0::Constant>(node)) {
                return true;
            }
            if (ov::as_type_ptr<ov::op::v0::Convert>(node) && ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0))) {
                return true;
            }
            return ov::as_type_ptr<ov::op::v1::Transpose>(node) && ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0));
        };
        auto is_placeholder = [](const std::shared_ptr<ov::Node> node) {
            return ov::as_type_ptr<op::Placeholder>(node);
        };

        const auto& fc = ov::as_type_ptr<op::FullyConnectedCompressed>(output.get_node_shared_ptr());
        const auto& input = fc->get_input_node_shared_ptr(0);
        if (!fc->get_input_partial_shape(0).is_dynamic()) {
            return false;
        }
        size_t user_fc_count = 0;
        int32_t nodes_with_bias = 0;
        int32_t nodes_with_zp = 0;
        for (const auto& u : input->get_users()) {
            const auto& fc_user = ov::as_type_ptr<op::FullyConnectedCompressed>(u);
            if (!fc_user) {
                continue;
            }

            // Skip horizontal fusion when the weight is not a constant. The fused-weight Concat
            // created during fusion relies on constant-folding at compile time; with a non-constant
            // weight (e.g. weights provided as runtime inputs) it cannot fold, survives to program
            // build, and may hit formats/types without a Concat implementation. Even when a Concat
            // impl exists, concatenating weights on every inference is pure overhead with no benefit.
            if (!is_constant(fc_user->get_input_node_shared_ptr(1))) {
                return false;
            }

            auto num_inputs = fc_user->inputs().size();
            if (num_inputs >= 5) {
                nodes_with_zp++;
            }
            for (size_t i = 2; i < num_inputs; ++i) {
                const auto& fc_input = fc_user->get_input_node_shared_ptr(i);
                if (!is_constant(fc_input) && !is_placeholder(fc_input)) {
                    return false;
                }
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
        ov::OutputVector weight_nodes;
        ov::OutputVector scale_nodes;
        ov::OutputVector bias_nodes;
        ov::OutputVector zp_nodes;
        int32_t bias_rank = -1;
        for (auto user : input_node->get_users()) {
            auto fc_user = ov::as_type_ptr<op::FullyConnectedCompressed>(user);
            if (fc_user) {
                OPENVINO_ASSERT(fc_user->inputs().size() >= 4, "Compressed FC should have at least 4 inputs");
                fc_nodes.push_back(fc_user);
                fc_nodes_vec.push_back(fc_user);
                weight_nodes.push_back(fc_user->input_value(1));
                if (!ov::as_type_ptr<op::Placeholder>(fc_user->get_input_node_shared_ptr(2))) {
                    if (bias_rank == -1) {
                        bias_rank = static_cast<int32_t>(fc_user->get_input_partial_shape(2).size());
                    }
                    if (bias_rank != static_cast<int32_t>(fc_user->get_input_partial_shape(2).size())) {
                        return false;
                    }
                    bias_nodes.push_back(fc_user->input_value(2));
                }
                scale_nodes.push_back(fc_user->input_value(3));
                if (fc_user->inputs().size() > 4) {
                    zp_nodes.push_back(fc_user->input_value(4));
                }
            }
        }
        // Weight layout depends on transpose_b: [N, K] when transpose_b=true (default),
        // [K, N] when transpose_b=false.
        const size_t weight_idx = 1;
        if (fc_nodes[0]->get_input_shape(weight_idx).size() != 2) {
            return false;
        }
        const bool transpose_b = fc_nodes[0]->get_transpose_b();
        // n_axis is the output (N) dimension, k_axis the contraction (K) dimension.
        const size_t n_axis = transpose_b ? 0 : 1;
        const size_t k_axis = transpose_b ? 1 : 0;
        auto weight_dtype = fc_nodes[0]->get_input_element_type(weight_idx);
        auto k_size = fc_nodes[0]->get_input_shape(weight_idx)[k_axis];
        std::vector<int64_t> orig_n_sizes;
        // merge weights, scale, zp
        for (auto fc : fc_nodes) {
            if (k_size != fc->get_input_shape(weight_idx)[k_axis]) {
                return false;
            }
            if (weight_dtype != fc->get_input_element_type(weight_idx)) {
                return false;
            }
            if (transpose_b != fc->get_transpose_b()) {
                return false;
            }
            orig_n_sizes.push_back(fc->get_input_shape(weight_idx)[n_axis]);
        }
        // Concatenates and folds the given constant-like nodes into a Constant right away, instead of relying on ConstantFolding pass:
        // unfolded Concat on weights (e.g. in case of unsupported precision) may then fail the pipeline if the GPU also doesn't support this.
        // So if the concatenation at ov::Model is not possible, the transformation just returns false
        auto concat_and_fold = [](const ov::OutputVector& outputs, int64_t axis, std::string_view name_suffix) -> std::shared_ptr<ov::Node> {
            auto concat = std::make_shared<ov::op::v0::Concat>(outputs, axis);
            auto folded = ov::util::get_constant_from_source(concat);
            if (!folded) {
                GPU_DEBUG_TRACE_DETAIL << "FullyConnectedHorizontalFusion: failed to constant-fold the concat of " << name_suffix << std::endl;
                return nullptr;
            }
            const auto nodes = ov::as_node_vector(outputs);
            folded->set_friendly_name(nodes[0]->get_friendly_name() + std::string(name_suffix));
            ov::copy_runtime_info(nodes, folded);
            return folded;
        };

        // Fold weights and scales before the graph is modified below, so that bailing out leaves the model intact
        auto fused_weight = concat_and_fold(weight_nodes, transpose_b ? 0 : 1, "_fused_weight");
        if (!fused_weight) {
            return false;
        }

        auto fused_scale = concat_and_fold(scale_nodes, 0, "_fused_scale");
        if (!fused_scale) {
            return false;
        }

        std::shared_ptr<ov::Node> fused_zps;
        if (!zp_nodes.empty()) {
            bool single_zp_value = (ov::shape_size(zp_nodes[0].get_shape()) == 1);
            int32_t scalar_zp_val = 0;
            if (single_zp_value) {
                if (auto zp_const = ov::as_type_ptr<ov::op::v0::Constant>(zp_nodes[0].get_node_shared_ptr())) {
                    scalar_zp_val = zp_const->cast_vector<int32_t>()[0];
                } else if (auto zp_convert = ov::as_type_ptr<ov::op::v0::Convert>(zp_nodes[0].get_node_shared_ptr())) {
                    auto zp_const = ov::as_type_ptr<ov::op::v0::Constant>(zp_convert->get_input_node_shared_ptr(0));
                    scalar_zp_val = zp_const->cast_vector<int32_t>()[0];
                }
                fused_zps = zp_nodes[0].get_node_shared_ptr();
            }
            if (single_zp_value) {
                for (size_t i = 1; i < zp_nodes.size(); ++i) {
                    bool current_is_scalar = (ov::shape_size(zp_nodes[i].get_shape()) == 1);
                    if (!current_is_scalar) {
                        return false;
                    }
                    // validate all zp values are same
                    int32_t cur_zp_val = 0;
                    if (auto zp_const = ov::as_type_ptr<ov::op::v0::Constant>(zp_nodes[i].get_node_shared_ptr())) {
                        cur_zp_val = zp_const->cast_vector<int32_t>()[0];
                    } else if (auto zp_convert = ov::as_type_ptr<ov::op::v0::Convert>(zp_nodes[i].get_node_shared_ptr())) {
                        auto zp_const = ov::as_type_ptr<ov::op::v0::Constant>(zp_convert->get_input_node_shared_ptr(0));
                        cur_zp_val = zp_const->cast_vector<int32_t>()[0];
                    } else {
                        OPENVINO_THROW("Unsupported zp input node for FC horizontal fusion");
                    }
                    if (cur_zp_val != scalar_zp_val) {
                        return false;
                    }
                }
            } else {
                fused_zps = concat_and_fold(zp_nodes, 0, "_fused_zps");
                if (!fused_zps) {
                    return false;
                }
            }
        }

        // Biases are processed last, as only here the graph gets modified: all the checks that may decline the fusion are done by now
        // check if the FCs do not have bias inputs, but all of the fc has a bias add user, set them as bias inputs
        // Currently horizontal fusing is applied only when fusing is applied for N dim
        // Also, fuse biases for the last dimension too, if
        // - Biases are constant
        // - Rank of the bias shapes are same
        // - all other dims except last dim is 1 (e.g., [1, 1, N])
        size_t n_bias_users = 0;
        bool bias_from_add_users = false;
        if (bias_nodes.empty()) {
            for (auto fc : fc_nodes) {
                if (fc->get_users().size() == 1 && fc->get_users()[0]->get_type_info() == ov::opset1::Add::get_type_info_static() &&
                    ov::is_type<ov::op::v0::Constant>(fc->get_users()[0]->inputs()[1].get_source_output().get_node())) {
                    auto bias_input1_shape = fc->get_users()[0]->get_input_partial_shape(1).get_shape();
                    if (bias_rank == -1) {
                        bias_rank = static_cast<int32_t>(bias_input1_shape.size());
                    }
                    if (bias_rank != static_cast<int32_t>(bias_input1_shape.size())) {
                        break;
                    }
                    size_t ndim_size = bias_input1_shape.back();
                    // allow only [1, 1, N] shape bias
                    if (std::accumulate(bias_input1_shape.begin(), bias_input1_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) != ndim_size) {
                        break;
                    }
                    n_bias_users++;
                }
            }

            if (n_bias_users == fc_nodes.size()) {
                for (size_t i = 0; i < fc_nodes.size(); ++i) {
                    auto orig_fc = fc_nodes[i];
                    auto bias_node = orig_fc->get_users()[0];
                    bias_nodes.push_back(bias_node->input_value(1));
                }
                bias_from_add_users = true;
            }
        }

        std::shared_ptr<ov::Node> fused_bias;
        if (bias_nodes.size() == fc_nodes.size()) {
            fused_bias = concat_and_fold(bias_nodes, bias_rank - 1, "_fused_bias");
            if (!fused_bias) {
                return false;
            }
        } else {
            fused_bias = std::make_shared<op::Placeholder>();
        }

        if (bias_from_add_users) {
            for (size_t i = 0; i < fc_nodes.size(); ++i) {
                auto orig_fc = fc_nodes[i];
                auto bias_node = orig_fc->get_users()[0];
                GPU_DEBUG_TRACE_DETAIL << "Set Add op user " << bias_node->get_friendly_name() << " as the FC " << orig_fc->get_friendly_name()
                                       << "'s bias input" << std::endl;
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
                fc_nodes[i] = ov::as_type_ptr<op::FullyConnectedCompressed>(new_fc);
                bias_node->clear_control_dependencies();
                orig_fc->clear_control_dependencies();
            }
        }

        // Create new fc with merged weights, bias, scale, zp
        std::shared_ptr<ov::Node> new_fc;
        if (fused_zps) {
            new_fc = std::make_shared<op::FullyConnectedCompressed>(input_node,
                                                                    fused_weight,
                                                                    fused_bias,
                                                                    fused_scale,
                                                                    fused_zps,
                                                                    fc_nodes[0]->get_output_type(),
                                                                    transpose_b);
        } else {
            new_fc =
                std::make_shared<op::FullyConnectedCompressed>(input_node, fused_weight, fused_bias, fused_scale, fc_nodes[0]->get_output_type(), transpose_b);
        }

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

        // Merge scalar multiply layers into one when all scalar constants have the same value.
        //
        //          FusedFC                     FusedFC
        //             |                           |
        //       VariadicSplit      ==>         new_Mul  (to be fused with FusedFC)
        //      /      |      \                    |
        //    Mul     Mul     Mul            VariadicSplit
        //     |       |       |             |     |     |
        const auto is_scalar_const = [](const ov::Output<ov::Node>& output) -> bool {
            if (!ov::is_type<ov::op::v0::Constant>(output.get_node())) {
                return false;
            }
            const auto shape = output.get_partial_shape();
            if (shape.is_dynamic()) {
                return false;
            }
            return ov::shape_size(shape.to_shape()) == 1;
        };

        std::vector<float> const_values;
        bool can_be_merged = true;
        std::shared_ptr<ov::op::v0::Constant> const_node = nullptr;
        for (auto& output : output_split->outputs()) {
            if (output.get_target_inputs().size() != 1) {
                can_be_merged = false;
                break;
            }
            auto* target_node = output.get_target_inputs().begin()->get_node();
            if (!ov::is_type<ov::op::v1::Multiply>(target_node)) {
                can_be_merged = false;
                break;
            }

            for (auto& input : target_node->inputs()) {
                if (input.get_source_output() != output) {
                    if (is_scalar_const(input.get_source_output())) {
                        const_node = ov::as_type_ptr<ov::op::v0::Constant>(input.get_source_output().get_node_shared_ptr());
                        const_values.emplace_back(const_node->cast_vector<float>()[0]);
                    } else {
                        can_be_merged = false;
                        break;
                    }
                }
            }
        }

        if (const_values.size() != split_size || !std::equal(const_values.begin() + 1, const_values.end(), const_values.begin())) {
            can_be_merged = false;
        }

        if (can_be_merged) {
            auto new_mul = std::make_shared<ov::op::v1::Multiply>(new_fc, const_node);
            new_mul->set_friendly_name(new_fc->get_friendly_name() + "_mul");
            ov::NodeVector fused_mul_nodes;
            output_split->input(0).replace_source_output(new_mul);
            for (auto& output : output_split->outputs()) {
                auto* target_node = output.get_target_inputs().begin()->get_node();
                fused_mul_nodes.push_back(target_node->shared_from_this());
                ov::replace_output_update_name(target_node->output(0), output);
            }
            ov::copy_runtime_info(fused_mul_nodes, new_mul);
        }

        GPU_DEBUG_TRACE_DETAIL << "Created a new fused FC " << new_fc_name << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(target_fc, "FullyConnectedHorizontalFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
