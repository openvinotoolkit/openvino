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
#include "binary_convolution_inst.h"
#include <vector>
#include <list>
#include <utility>

using namespace cldnn;

remove_redundant_reorders::remove_redundant_reorders(layout_optimizer& lo_ref, bool enable_reorder_fusing, bool update_implementations)
    : base_pass("remove_redundant_reorders"), lo(lo_ref), enable_reorder_fusing(enable_reorder_fusing), update_implementations(update_implementations) {}

void remove_redundant_reorders::run(program_impl& p) {
    auto update_implementation = [&](program_node& node) {
        if (!update_implementations)
            return;

        auto& eng = p.get_engine();
        auto new_impl = node.type()->choose_impl(eng, node);
        node.set_selected_impl(std::move(new_impl));
    };

    // Fuse reorders into primitives
    auto itr = p.get_processing_order().begin();
    if (enable_reorder_fusing) {
        while (itr != p.get_processing_order().end()) {
            auto node_ptr = *itr++;
            if (!node_ptr->is_type<reorder>())  // only care for reorders
                continue;

            auto& node = node_ptr->as<reorder>();

            auto& input = node.input();
            auto output_layout = node.get_output_layout();

            if (node.is_output())
                continue;

            if (node.has_mean() || !node.get_primitive()->subtract_per_feature.empty())
                continue;

            if (!node.get_fused_activations_funcs().empty())
                continue;

            auto same_data_type = input.get_output_layout().data_type == output_layout.data_type;
            if (!same_data_type)
                continue;

            bool all_users_fuse = true;
            std::vector<program_node*> recalc_list;

            for (auto usr : node.get_users()) {
                if (!lo.can_fuse_reorder(input, *usr, input.get_output_layout().format, usr->get_output_layout().format)) {
                    all_users_fuse = false;
                    break;
                }

                if (usr->is_type<fully_connected>())
                    recalc_list.push_back(usr);
            }

            if (!all_users_fuse)
                continue;

            auto output_padded = static_cast<bool>(output_layout.data_padding);
            auto can_omit_padding = output_layout.format == format::b_fs_yx_fsv16 && input.get_output_layout().format == format::bfyx;

            if (output_padded && !can_omit_padding) {
                if (input.get_users().size() != 1)
                    continue;

                if (input.is_type<input_layout>())
                    continue;

                input.merge_output_padding(output_layout.data_padding);
            }

            node.can_be_optimized(true);
            p.extract_and_remove(node);

            for (auto rl : recalc_list) {
                rl->recalc_output_layout(true);
            }
        }
    }

    // Shrink reorder chains
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node = *itr++;
        if (!node->is_type<reorder>())  // only care for reorders
            continue;

        auto& r_node = node->as<reorder>();
        auto& dep_node = r_node.get_dependency(0);

        if (!dep_node.is_type<reorder>())
            continue;

        auto& r_dep_node = dep_node.as<reorder>();

        bool remove_dep = r_dep_node.get_users().size() == 1 &&
            !r_dep_node.has_mean() &&
            r_dep_node.get_primitive()->subtract_per_feature.empty() &&
            !r_dep_node.is_output() &&
            r_dep_node.get_fused_activations_funcs().empty();

        bool remove_current =
            r_dep_node.get_users().size() == 1 &&
            !r_dep_node.is_output() &&
            !r_node.has_mean() &&
            r_node.get_primitive()->subtract_per_feature.empty() &&
            r_node.get_fused_activations_funcs().empty();

        if (remove_dep) {
            r_dep_node.can_be_optimized(true);
            p.add_optimized_primitive_info(r_dep_node.id());
            p.extract_and_remove(r_dep_node);
            update_implementation(r_node);
        } else if (remove_current) {
            auto output_layout = r_node.get_output_layout();
            auto dep_prim = std::const_pointer_cast<reorder>(r_dep_node.get_primitive());
            dep_prim->output_format = output_layout.format;
            dep_prim->output_data_type = output_layout.data_type;

            r_node.can_be_optimized(true);
            p.add_optimized_primitive_info(r_node.id());
            p.extract_and_remove(r_node);

            r_dep_node.recalc_output_layout(false);
            update_implementation(r_dep_node);
        }
    }

    // Optimize reorders not changing memory layout
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node = *itr++;
        if (!node->is_type<reorder>())  // only care for reorders
            continue;

        auto& r_node = node->as<reorder>();

        if (r_node.has_mean() ||
            !r_node.get_primitive()->subtract_per_feature.empty() ||
            r_node.is_output() ||
            !r_node.get_fused_activations_funcs().empty())
            continue;

        auto o_layout = r_node.get_output_layout();
        auto i_layout = r_node.get_dependency(0).get_output_layout();

        // Optimize reorder b_fs_yx_fsv16 -> bfyx when spatials are equal to 1. In this case we can reinterpret buffer,
        // but pads need to be handled correctly.
        if (i_layout.format == format::b_fs_yx_fsv16 && o_layout.format == format::bfyx &&
            i_layout.size.spatial[0] == 1 && i_layout.size.spatial[1] == 1 &&
            o_layout.data_padding.upper_size() == (tensor)0 && o_layout.data_padding.lower_size() == (tensor)0) {
            r_node.can_be_optimized(true);
            if (i_layout.size.feature[0] % 16 != 0) {
                auto pad_lo = o_layout.data_padding.lower_size();
                auto pad_hi = o_layout.data_padding.upper_size();
                pad_hi.feature[0] = i_layout.size.feature[0] % 16;
                r_node.merge_output_padding(padding{pad_lo.sizes(), pad_hi.sizes()});
            }
            continue;
        }

        auto ident = program_helpers::are_layouts_identical(o_layout, i_layout);

        if (!ident.second)
            continue;

        // mark as optimized
        r_node.can_be_optimized(true);
        r_node.requires_reinterpret(!ident.first);
        if (ident.first) {  // no need of reshape
            p.add_optimized_primitive_info(r_node.get_primitive()->id);
            p.extract_and_remove(
                r_node);  // try to remove if possible (with respect to r_node not being marked as output)
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
                user->get_fused_activations_funcs().empty()) {
                auto l1 = node->get_output_layout();
                auto l2 = user->get_output_layout();

                auto ident = program_helpers::are_layouts_identical(l1, l2);
                if (ident.first)
                    r_nodes_to_remove.push_back(user);
            }
        }

        if (r_nodes_to_remove.empty())
            continue;

        if (itr == p.get_processing_order().end())
            break;

        auto rem_itr = r_nodes_to_remove.begin();
        while (rem_itr != r_nodes_to_remove.end()) {
            auto remove_reorder_node = *rem_itr++;
            // Outer loop iterator has been already moved, so if we try to remove a node which the iterator
            // pointing to, we should increment it again
            if (remove_reorder_node == *itr)
                itr++;
            p.replace_all_usages(*remove_reorder_node, *node);
            p.add_optimized_primitive_info(remove_reorder_node->id());
            p.remove_all_connections(*remove_reorder_node);
            p.remove_if_dangling(*remove_reorder_node);
        }
    }

    // This pass removed reorder if previous node can store directly to required layout
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>())  // only care for reorders
            continue;

        auto& node = node_ptr->as<reorder>();

        auto& input = node.input();
        auto output_layout = node.get_output_layout();

        if (node.is_output())
            continue;

        if (node.has_mean() || !node.get_primitive()->subtract_per_feature.empty())
            continue;

        if (!node.get_fused_activations_funcs().empty())
            continue;

        if (input.get_users().size() != 1 || node.get_users().empty())
            continue;

        auto same_data_type = input.get_output_layout().data_type == output_layout.data_type;
        if (!same_data_type)
            continue;

        if (!lo.can_fuse_reorder_to_prev(input, *node.get_users().front(), input.get_output_layout().format, output_layout.format))
            continue;

        input.set_output_layout(output_layout, false);
        if (input.type()->does_possible_implementation_exist(p.get_engine(), input)) {
            p.replace_all_usages(node, input);
            p.add_optimized_primitive_info(node.id());
            p.remove_all_connections(node);
            p.remove_if_dangling(node);
        }
    }

    // This pass removed reorder if the next node supports reorder's input format
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<reorder>() || !node->is_in_data_flow() || node->get_users().size() != 1 || node->get_dependencies().size() != 1)
            continue;

        auto& usr = node->get_users().front();
        auto& dep = node->get_dependency(0);
        if (!usr->is_type<quantize>() || node->get_output_layout().format != format::bfyx ||
            (dep.get_output_layout().format != format::b_fs_yx_fsv16 && dep.get_output_layout().format != format::fs_b_yx_fsv32))
            continue;

        dep.merge_output_padding(node->get_output_layout().data_padding);
        p.replace_all_usages(*node, dep);
        p.add_optimized_primitive_info(node->id());
        p.remove_all_connections(*node);
        p.remove_if_dangling(*node);
    }

    // This pass removes reorder for Convolution BFYX -> FS_B_YX_FSV32
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<reorder>() || !node->is_in_data_flow() || node->get_users().size() != 1 || node->get_dependencies().size() != 1)
            continue;

        auto& usr = node->get_users().front();
        auto& dep = node->get_dependency(0);
        if (!(usr->is_type<convolution>()) ||
             (usr->get_output_layout().data_type != dep.get_output_layout().data_type) ||
             (usr->get_output_layout().format != format::fs_b_yx_fsv32) ||
             (dep.get_output_layout().format != format::bfyx))
            continue;

        if (dep.is_type<input_layout>())
            continue;

        if (usr->as<convolution>().get_primitive()->groups != 1)
            continue;

        dep.merge_output_padding(node->get_output_layout().data_padding);
        p.replace_all_usages(*node, dep);
        p.get_processing_order().erase(node);
        p.add_optimized_primitive_info(node->id());
        p.remove_all_connections(*node);
        p.remove_if_dangling(*node);
    }
}
