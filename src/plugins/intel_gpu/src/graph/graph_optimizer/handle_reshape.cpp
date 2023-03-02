// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "reshape_inst.h"

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
            if (out_node->is_type<reshape>())
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

            // find the users of reshape that are reorder type, if none present then skip the current node
            for (const auto& user : node->get_users()) {
                if (user->is_type<reorder>())
                    reorder_node_to_split.push_back(user);
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
                        auto& new_reshape_node = p.get_or_create(new_reshape);
                        user->replace_dependency(0, input_node);
                        p.add_intermediate(new_reshape_node, *user, 0);
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
                    auto format = cldnn::format::bfyx;
                    if (dims == 5)
                        format = cldnn::format::bfzyx;
                    else if (dims == 6)
                        format = cldnn::format::bfwzyx;
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
