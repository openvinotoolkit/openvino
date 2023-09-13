// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "program_helpers.h"
#include "reshape_inst.h"
#include "layout_optimizer.h"

#include "gemm_inst.h"
#include "pooling_inst.h"
#include "fully_connected_inst.h"

#include <iterator>
#include <vector>
#include <memory>

using namespace cldnn;

// reshape primitive by definition does not change underlying data, only shape description
// however during graph initialization and data optimization the layouts can be changed without user's knowledge,
// when reshape is followed by reorder, it is likely that reorder's output will not be as expected (for example reshape
// with flattened shape) this pass resolved the issue by changing graph in the following way
//- in case reshape has multiple users with reshape->reorder sequence, it will be splitted to multiple reshape
// primitives with single user
//- in case of reshape->reorder sequence, the additional reorder before reshape will be added,
//  if last reorder does not contain padding or mean subtract, it will be removed later in the graph
void handle_reshape::run(program& p) {
    // Remove reshapes that don't change the layout of output
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto node = (*node_itr++);
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            auto& input_node = node.input();
            auto input_lay = input_node.get_output_layout();
            auto output_lay = node.get_output_layout();

            if (!node.is_in_place() ||
                node.has_fused_primitives())
                return;

            if (input_lay.identical(output_lay)) {
                p.add_optimized_primitive_info(node.id());
                p.extract_and_remove(node);
            } else if (input_lay.compatible(output_lay) && input_node.is_type<data>()) {
                input_node.set_output_layout(output_lay, false);
                p.add_optimized_primitive_info(node.id());
                p.extract_and_remove(node);
            } else if (input_lay.compatible(output_lay)) {
                p.add_optimized_primitive_info(node.id());
                node.can_be_optimized(true);
            }
        });
    }
    // If graph contains sequence of reshape nodes, we can remove all except the last one
    // E.g. pattern permute+flatten+reshape (common for object detection topologies) is represented as
    // permute+reshape+reshape in cldnn and can be simplified to permute+reshape.
    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            if (node.is_output() || node.get_users().size() > 1 || node.has_fused_primitives() || node.is_dynamic())
                return;

            auto& out_node = node.get_users().front();

            if (!out_node->is_type<reshape>())
                return;

            const auto& out_reshape = out_node->as<reshape>();
            // In case of new shape infer we should not shrink reshapes chain if first reshape changes input rank, e.g.
            // [a, b] -> reshape1 -> [a1, b1, c1] -> reshape2 -> [a2, b2, 0] and any of the reshapes has special_zero=true
            // Configuration above will fail if we remove reshape1 node as attempt to handle special zero will fail due to small rank of input
            if (p.get_config().get_property(ov::intel_gpu::allow_new_shape_infer) &&
                out_node->get_output_pshape().size() != node.get_input_pshape().size() &&
                (out_reshape.get_primitive()->special_zero || node.get_primitive()->special_zero))
                return;

            p.extract_and_remove(node);
        });
    }

    for (const auto& node : p.get_processing_order()) {
        if (node->is_type<reshape>()) {
            auto& input_node = node->get_dependency(0);

            if (input_node.is_type<reorder>())
                continue;

            node->get_output_layout();

            // vector for storing nodes that are reorder type, for which splitted primitives are needed (except for the
            // first one where orginal reshape will be used)
            std::vector<program_node*> reorder_node_to_split;
            std::vector<program_node*> onednn_users;

            // find the users of reshape that are reorder type, if none present then skip the current node
            // find users who are onednn impl
            for (const auto& user : node->get_users()) {
                if (user->is_type<reorder>())
                    reorder_node_to_split.push_back(user);
                if (user->get_preferred_impl_type() == cldnn::impl_types::onednn)
                    onednn_users.push_back(user);
            }

            // If onednn user doesn't support new input data type from future "reorder:_reshape_input_" reorder,
            // remove target reorder_node to keep original datatype
            if (!onednn_users.empty() && !reorder_node_to_split.empty()) {
                // Copy reorder_node_to_split to iteration
                std::vector<program_node*> reorder_users(reorder_node_to_split);
                for (const auto& reorder_node : reorder_users) {
                    auto output_data_type = reorder_node->get_output_layout().data_type;
                    bool onednn_support = true;
                    for (const auto& user : onednn_users) {
                        auto out_dt = user->get_output_layout().data_type;
                        if (user->is_type<fully_connected>() || user->is_type<gemm>()) {
                            bool is_fc = user->is_type<fully_connected>();
                            auto wei_dt = is_fc ? user->as<fully_connected>().weights().get_output_layout().data_type :
                                                    user->as<gemm>().get_input_layout(1).data_type;
                            onednn_support = layout_optimizer::onednn_check_data_types_for_fc_gemm(output_data_type, wei_dt, out_dt);
                        } else if (user->is_type<convolution>() || user->is_type<deconvolution>()) {
                            bool is_conv = user->is_type<convolution>();
                            auto wei_dt = is_conv ? user->as<convolution>().weights().get_output_layout().data_type :
                                                    user->as<deconvolution>().weights().get_output_layout().data_type;
                            onednn_support = layout_optimizer::onednn_check_data_types_for_convolution(output_data_type, wei_dt, out_dt);
                        } else if (user->is_type<pooling>()) {
                            onednn_support = layout_optimizer::onednn_check_data_types_for_pooling(output_data_type, out_dt);
                        }

                        if (!onednn_support) {
                            reorder_node_to_split.erase(std::remove(reorder_node_to_split.begin(), reorder_node_to_split.end(), reorder_node),
                                                        reorder_node_to_split.end());
                            break;
                        }
                    }
                }
            }

            if (!reorder_node_to_split.empty()) {
                auto& prim_node = node->as<reshape>();
                const auto& prim = prim_node.get_primitive();
                auto output_shape = prim->output_shape;

                // vector for storing reshape nodes to connect to new reorder nodes (if needed)
                std::vector<program_node*> reorder_reshape_nodes;

                bool found_one = false;
                auto reshape_users = node->get_users();
                for (const auto& user : reshape_users) {
                    // reshape node for first user will be the orginal reshape from the graph
                    if (!found_one) {
                        if ((std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) !=
                            reorder_node_to_split.end()) && (user->get_output_layout().get_rank() == node->get_output_layout().get_rank()))
                            reorder_reshape_nodes.push_back(node);
                        found_one = true;
                        continue;
                    }

                    // other reshapes will be clones of the orginal one connected to reshape->reorder sequences
                    if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) !=
                        reorder_node_to_split.end()) {
                        auto new_reshape = std::make_shared<reshape>("reorder:_reshape_split_" + user->id() + "_" + node->id(),
                                                                     input_node.id(),
                                                                     output_shape);
                        new_reshape->special_zero = prim->special_zero;
                        new_reshape->output_partial_shape = prim->output_partial_shape;
                        new_reshape->output_pattern = prim->output_pattern;
                        new_reshape->mode = prim->mode;
                        new_reshape->input = prim->input;
                        auto& new_reshape_node = p.get_or_create(new_reshape);
                        user->replace_dependency(0, input_node);
                        p.add_intermediate(new_reshape_node, *user, 0);
                        if (new_reshape->input_size() == 2) {
                            p.add_connection(prim_node.get_dependency(1), new_reshape_node);
                        }

                        reorder_reshape_nodes.push_back(&new_reshape_node);
                    }
                }

                if (reorder_reshape_nodes.size() == 0)
                    continue;

                // add new reorder nodes to proper reshape node
                auto reshape_reorder_id = 0;
                for (const auto& reorder_node : reorder_node_to_split) {
                    auto& reorder_reshape_node = reorder_reshape_nodes[reshape_reorder_id];
                    auto reshape_in_layout = reorder_node->get_output_layout();
                    auto dims = cldnn::format::dimension(reshape_in_layout.format);
                    auto format = cldnn::format::get_default_format(dims);
                    auto reshape_input = std::make_shared<reorder>(
                        "reorder:_reshape_input_" + reorder_node->id() + "_" + reorder_reshape_node->id(),
                        input_node.id(),
                        format,
                        reshape_in_layout.data_type);
                    auto& reshape_input_node = p.get_or_create(reshape_input);
                    p.add_intermediate(reshape_input_node,
                                       *reorder_reshape_node,
                                       0,
                                       reshape_input_node.get_dependencies().empty());
                    reshape_reorder_id++;
                    reshape_input_node.recalc_output_layout();
                }
            }

            auto reshape_layout = node->get_output_layout();
            auto target_format = format::get_default_format(reshape_layout.get_rank());

            if (!(node->is_output()) && (reshape_layout.format != target_format)) {
                auto target_layout = layout({reshape_layout.get_partial_shape(), reshape_layout.data_type, target_format});
                // when some primitive does an implicit reorder to some other format then we lose the info about pitches
                // in reshape stage we assume user provides the input vector in bfyx
                if (!reshape_layout.compatible(target_layout)) {
                    auto reshape_input = std::make_shared<reorder>("reorder:_reshape_input_" + node->id(),
                                                                   input_node.id(),
                                                                   target_format,
                                                                   reshape_layout.data_type);
                    auto& reshape_input_node = p.get_or_create(reshape_input);
                    p.add_intermediate(reshape_input_node, *node, 0, reshape_input_node.get_dependencies().empty());
                    reshape_input_node.recalc_output_layout();

                    auto reshape_users = node->get_users();
                    for (const auto& user : reshape_users) {
                        auto reshape_output = std::make_shared<reorder>("reorder:_reshape_output_" + node->id(),
                                                                        user->id(),
                                                                        reshape_layout.format,
                                                                        reshape_layout.data_type);
                        auto& reshape_output_node = p.get_or_create(reshape_output);
                        p.add_intermediate(reshape_output_node,
                                           *user,
                                           *node,
                                           reshape_output_node.get_dependencies().empty());
                        reshape_output_node.recalc_output_layout();
                    }
                }
            }
        }
    }
}
