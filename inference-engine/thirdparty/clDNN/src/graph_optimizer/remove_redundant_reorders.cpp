/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_helpers.h"
#include <vector>
#include <list>

using namespace cldnn;

remove_redundant_reorders::remove_redundant_reorders(bool bfyx_to_bfyx_f16_opt)
    : base_pass("remove_redundant_reorders"), bfyx_to_bfyx_f16_opt(bfyx_to_bfyx_f16_opt) {}

void remove_redundant_reorders::run(program_impl& p) {
    auto itr = p.get_processing_order()
                   .begin();  // note we need to use iterators since currently processed element can be removed
    while (itr != p.get_processing_order().end()) {
        auto& node = (*itr++);          // post-inc to avoid invalidation due to possible erase
        if (!node->is_type<reorder>())  // only care for reorders
            continue;

        program_node* current_node = node;
        std::vector<program_node*> r_nodes_to_remove;

        auto optimize = true;
        while (current_node) {
            auto& r_node = current_node->as<reorder>();
            current_node = nullptr;

            if (r_node.has_mean() ||
                !r_node.get_primitive()->subtract_per_feature.empty() ||  // do not optimize if mean of subtract are present
                r_node.is_output() ||                   // do not optimize when both reorder and layer before are outputs
                r_node.get_fused_activation_func() != activation_none) {
                // TODO Verify whether optimization can be performed at current sub-chain of reorders
                optimize = false;
                break;
            }

            r_nodes_to_remove.push_back(&r_node);

            if (r_node.get_dependency(0).is_type<reorder>() && r_node.get_dependencies().size() == 1 &&
                r_node.get_users().size() == 1 && r_node.get_dependency(0).get_users().size() == 1)
                current_node = &r_node.get_dependency(0);
        }
        if (!optimize)
            continue;

        assert(node->get_dependencies().size() == 1 &&
               "reorder without mean should have exactly one dependecy (input)");
        auto& r_output = r_nodes_to_remove.front();
        auto& r_input = r_nodes_to_remove.back()->get_dependency(0);
        auto o_layout = r_output->get_output_layout();
        auto i_layout = r_input.get_output_layout();

        auto ident = program_helpers::are_layouts_identical(o_layout, i_layout);
        if (!ident.second)
            continue;

        for (auto remove_reorder_node : r_nodes_to_remove) {
            auto& r_node = remove_reorder_node->as<reorder>();

            if (ident.first && ident.second && r_node.is_output() &&
                r_node.get_dependency(0).is_input()) {  // do not optimize when reorder is output and layer before is input
                optimize = false;
                break;
            }
        }
        if (!optimize)
            continue;

        auto rem_itr = r_nodes_to_remove.begin();
        while (rem_itr != r_nodes_to_remove.end()) {
            auto remove_reorder_node = *rem_itr++;
            auto& r_node = remove_reorder_node->as<reorder>();
            // mark as optimized
            r_node.can_be_optimized(true);
            r_node.requires_reinterpret(!ident.first);
            if (ident.first) {  // no need of reshape
                p.add_optimized_primitive_info(r_node.get_primitive()->id);
                p.extract_and_remove(
                    r_node);  // try to remove if possible (with respect to r_node not being marked as output)
            }
        }
    }

    // This pass removes redundant reorders in case when several users of a node have the same input reorders
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<reorder>())
            continue;

        std::list<program_node*> r_nodes_to_remove;

        if (node->get_dependencies().size() != 1)
            continue;

        auto& dep = node->get_dependency(0);

        for (auto& user : dep.get_users()) {
            if (user->is_type<reorder>() &&
                user != node &&
                !user->is_output() &&
                user->get_fused_activation_func() == cldnn_activation_func_t::activation_none) {
                auto l1 = node->get_output_layout();
                auto l2 = user->get_output_layout();

                auto ident = program_helpers::are_layouts_identical(l1, l2);
                if (ident.first)
                    r_nodes_to_remove.push_back(user);
            }
        }

        if (r_nodes_to_remove.empty())
            continue;

        auto rem_itr = r_nodes_to_remove.begin();
        while (rem_itr != r_nodes_to_remove.end()) {
            auto remove_reorder_node = *rem_itr++;
            // Outer loop iterator has been already moved, so if we try to remove a node which the iterator
            // pointing to, we should increment it again
            if (remove_reorder_node == *itr)
                itr++;
            p.replace_all_usages(*remove_reorder_node, *node);
            p.get_processing_order().erase(remove_reorder_node);
            p.add_optimized_primitive_info(remove_reorder_node->id());
            p.remove_all_connections(*remove_reorder_node);
            p.remove_if_dangling(*remove_reorder_node);
        }
    }

    if (bfyx_to_bfyx_f16_opt) {
        // Removes reorder bfyx->bfyx_f16 when ic=3 and oc>=16 in order to enable specific kernel
        // Needs to be done after passes that can change layouts (like prepare_padding)
        itr = p.get_processing_order().begin();
        while (itr != p.get_processing_order().end()) {
            auto &node = *itr++;
            if (!node->is_type<reorder>())
                continue;

            if (node->get_dependencies().size() != 1 || node->get_users().size() != 1)
                continue;

            auto &user = node->get_users().front();
            auto &dep = node->get_dependency(0);

            if (user->is_type<convolution>() &&
                node->get_fused_activation_func() == cldnn_activation_func_t::activation_none &&
                dep.get_output_layout().format == format::bfyx &&
                dep.get_output_layout().size.feature[0] == 3 &&
                node->get_output_layout().format == format::bfyx_f16 &&
                user->get_output_layout().size.feature[0] >= 16) {
                p.extract_and_remove(*node);
            }
        }
    }
}
