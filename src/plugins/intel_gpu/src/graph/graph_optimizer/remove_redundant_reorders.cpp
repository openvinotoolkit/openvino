// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"

#include "pass_manager.h"
#include "program_helpers.h"

#include "reshape_inst.h"
#include "convert_color_inst.h"
#include "one_hot_inst.h"
#include "shape_of_inst.h"
#include "gather_inst.h"
#include "select_inst.h"
#include "eltwise_inst.h"
#include "broadcast_inst.h"
#include "permute_inst.h"
#include "depth_to_space_inst.h"
#include "concatenation_inst.h"
#include "region_yolo_inst.h"
#include "fully_connected_inst.h"
#include "mvn_inst.h"

#include <vector>
#include <list>
#include <utility>

using namespace cldnn;

#define LOG_NODE_REMOVAL(id)      GPU_DEBUG_LOG_PASS << __func__ << ":" << __LINE__  << ": remove node: " << (id) << std::endl;
#define LOG_NODE_REPLACEMENT(id)  GPU_DEBUG_LOG_PASS << __func__ << ":" << __LINE__  << ": replace node: " << (id) << std::endl;

namespace {

bool does_any_user_have_impl_type(program_node& node, impl_types impl) {
    for (auto& user : node.get_users()) {
        if (user->get_preferred_impl_type() == impl)
            return true;
    }

    return false;
}

}  // namespace

remove_redundant_reorders::remove_redundant_reorders(bool enable_reorder_fusing, bool update_implementations,
    bool remove_output_reorders)
    : base_pass("remove_redundant_reorders"), enable_reorder_fusing(enable_reorder_fusing), update_implementations(update_implementations),
    remove_output_reorders(remove_output_reorders) {}

void remove_redundant_reorders::run(program& p) {
    auto& lo = p.get_layout_optimizer();
    auto update_implementation = [&](program_node& node) {
        if (!update_implementations)
            return;

        node.set_unique_id();
        node.set_selected_impl(node.type()->create_impl(node));
        if (auto impl = node.get_selected_impl()) {
            auto params = node.get_kernel_impl_params();
            p.get_kernels_cache().add_kernels_source(*params, impl->get_kernels_source());
        }
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

            if (!node.is_simple_reorder() || node.is_output())
                continue;

            std::function<bool(program_node&)> has_quantize_user;
            has_quantize_user = [&has_quantize_user](program_node& node) -> bool {
                auto& users = node.get_users();
                if (users.size() != 1)
                    return false;
                if (users.front()->is_type<quantize>())
                    return true;
                if (users.front()->is_type<reorder>())
                    return has_quantize_user(*users.front());
                return false;
            };

            // Avoid different data types between input and output
            auto same_data_type = input.get_output_layout().data_type == output_layout.data_type;
            auto i8_u8_input = input.get_output_layout().data_type == data_types::i8 ||
                               input.get_output_layout().data_type == data_types::u8;
            auto quantize_user = has_quantize_user(node);

            if (!same_data_type && !(i8_u8_input && quantize_user))
                continue;

            // Avoid optimization of nv12 reorder
            if (node.get_dependencies().size() != 1 || node.get_primitive()->has_surface_input())
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
            auto can_omit_padding = ((output_layout.format == format::b_fs_yx_fsv16 || output_layout.format == format::b_fs_yx_fsv32) &&
                                    (input.get_output_layout().format == format::bfyx || input.get_output_layout().format == format::b_fs_yx_fsv4)) ||
                                    (output_layout.format == format::b_fs_zyx_fsv16 && input.get_output_layout().format == format::bfzyx);

            if (output_padded && !can_omit_padding) {
                if (input.get_users().size() != 1)
                    continue;

                if (input.is_type<input_layout>())
                    continue;

                input.merge_output_padding(output_layout.data_padding);
            }

            node.can_be_optimized(true);
            LOG_NODE_REMOVAL(node.id());
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

        // Do not opt out result reorder of Loop body network
        bool is_loop_body_network_output = (r_node.get_program().is_body_program() && r_node.is_output());
        if (is_loop_body_network_output)
            continue;

        auto& dep_node = r_node.get_dependency(0);

        if (!dep_node.is_type<reorder>())
            continue;

        auto& r_dep_node = dep_node.as<reorder>();

        bool remove_dep = r_dep_node.is_simple_reorder() &&
                          r_dep_node.get_users().size() == 1 &&
                          !r_dep_node.is_output() &&
                          !r_dep_node.get_primitive()->has_surface_input() &&
                          !r_node.get_primitive()->weights_reorder_params;

        // for chains like
        // fp32 -> reorder -> u8 -> reorder -> fp32
        // we can't fuse two reorder primitives as first one must do cast to u8 data type which changes the values
        if (!data_type_traits::is_floating_point(r_dep_node.get_output_layout().data_type) &&
            data_type_traits::is_floating_point(r_dep_node.get_input_layout().data_type)) {
            continue;
        }

        bool remove_current = r_node.is_simple_reorder() &&
                              r_dep_node.get_users().size() == 1 &&
                              !r_dep_node.is_output() &&
                              !r_node.get_primitive()->has_surface_input() &&
                              r_node.get_input_layout().data_padding == r_node.get_output_layout().data_padding;

        if (remove_dep) {
            // for chains like
            // b_fs_yx_fsv16 -> reorder(ofmt:bfyx) -> bfyx -> reorder(ofmt:any) -> bfyx
            // if output_format of current node is format::any, input format of the dependency node is propagated as it is
            // b_fs_yx_fsv16 -> reorder(ofmt:any) -> b_fs_yx_fsv16
            // so output format of dependency node must be stored in output_format of current node
            // b_fs_yx_fsv16 -> reorder(ofmt:bfyx) -> bfyx
            auto output_layout = r_dep_node.get_output_layout();
            auto prim = std::const_pointer_cast<reorder>(r_node.get_primitive());
            if (prim->output_format == format::any)
                prim->output_format = output_layout.format;

            LOG_NODE_REMOVAL(r_dep_node.id());
            r_dep_node.can_be_optimized(true);
            p.add_optimized_primitive_info(r_dep_node.id());
            p.extract_and_remove(r_dep_node);
            update_implementation(r_node);
        } else if (remove_current) {
            auto output_layout = r_node.get_output_layout();
            auto dep_prim = std::const_pointer_cast<reorder>(r_dep_node.get_primitive());
            dep_prim->output_format = output_layout.format;
            dep_prim->output_data_types = {output_layout.data_type};

            LOG_NODE_REMOVAL(r_node.id());
            r_node.can_be_optimized(true);
            p.add_optimized_primitive_info(r_node.id());
            p.extract_and_remove(r_node);

            r_dep_node.recalc_output_layout(false);
            update_implementation(r_dep_node);
        }
    }

    // Fuse reorder through concat in case of batched nv12 inputs to handle chains like:
    // Reorder (nv12_uint8 -> bfyx_uint8) \|
    // Reorder (nv12_uint8 -> bfyx_uint8)  -> Concat (uint8) -> Reorder (uint8 -> fp32)
    // Reorder (nv12_uint8 -> bfyx_uint8) /|
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node = *itr++;
        if (!node->is_type<reorder>())
            continue;

        auto& r_node = node->as<reorder>();
        if (!r_node.get_primitive()->has_surface_input() ||
            r_node.is_output() ||
            r_node.has_mean() ||
            r_node.get_users().size() > 1 ||
            r_node.get_primitive()->subtract_per_feature.size() ||
            r_node.has_fused_primitives())
            continue;

        if (!r_node.get_users().front()->is_type<concatenation>())
            continue;

        auto& concat_node = r_node.get_users().front()->as<concatenation>();
        if (concat_node.get_output_layout().batch() == 1)
            continue;

        if (!concat_node.get_users().front()->is_type<reorder>())
            continue;

        auto& r_node_next = concat_node.get_users().front()->as<reorder>();

        if (r_node.get_output_layout().data_type == r_node_next.get_output_layout().data_type)
            continue;

        auto new_layout = r_node.get_output_layout();
        new_layout.data_type = r_node_next.get_output_layout().data_type;

        auto orig_reorder_prim = r_node.get_primitive();
        auto new_reorder_prim = std::make_shared<reorder>(r_node.id() + "_fused",
            orig_reorder_prim->input[0],
            new_layout);
        new_reorder_prim->input_mem_type = orig_reorder_prim->input_mem_type;

        auto& new_reorder_node = p.get_or_create(new_reorder_prim);

        LOG_NODE_REPLACEMENT(r_node.id());
        p.replace(r_node, new_reorder_node);
        new_reorder_node.recalc_output_layout();
    }

    // Optimize reorders not changing memory layout
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node = *itr++;
        if (!node->is_type<reorder>())  // only care for reorders
            continue;

        if (node->is_dynamic())
            continue;

        auto& r_node = node->as<reorder>();

        bool no_output_optimization = remove_output_reorders ?
            r_node.is_output() && (r_node.get_dependency(0).is_output() || r_node.get_dependency(0).is_type<input_layout>() ||
                r_node.get_dependency(0).can_be_optimized() || r_node.get_dependency(0).get_users().size() != 1) : r_node.is_output();

        // Do not opt out result reorder of Loop body network
        no_output_optimization |= (r_node.get_program().is_body_program() && r_node.is_output());

        if (!r_node.is_simple_reorder() ||
            no_output_optimization ||
            r_node.get_primitive()->has_surface_input())
            continue;

        auto o_layout = r_node.get_output_layout();
        const auto& i_layout = r_node.get_input_layout(0);

        auto is_r_node_rank_changed = r_node.get_output_layout().get_rank() != r_node.get_dependency(0).get_output_layout().get_rank();
        if (is_r_node_rank_changed &&
            ((!update_implementations && r_node.get_dependency(0).is_type<crop>()) ||
             (r_node.get_dependency(0).is_type<crop>() && r_node.get_dependency(0).can_be_optimized())))
            continue;

        // Optimize reorder b_fs_yx_fsv16 -> bfyx when spatials are equal to 1. In this case we can reinterpret buffer,
        // but pads need to be handled correctly.
        if (i_layout.format == format::b_fs_yx_fsv16 && o_layout.format == format::bfyx && !r_node.is_output() &&
            i_layout.spatial(0) == 1 && i_layout.spatial(1) == 1 &&
            i_layout.data_padding._upper_size[2] == 0 && i_layout.data_padding._lower_size[2] == 0 &&
            i_layout.data_padding._upper_size[3] == 0 && i_layout.data_padding._lower_size[3] == 0 &&
            !o_layout.data_padding &&
            i_layout.data_type == o_layout.data_type &&
            !does_any_user_have_impl_type(r_node, impl_types::onednn)) {
            // If the newly aligned pad is merged into output layout during post_optimize_graph phase
            // and then buffer is reinterpreted, user node cannot handle pad properly for kernel execution
            if (!update_implementations || (i_layout.feature() % 16 == 0 &&
                !i_layout.data_padding && !o_layout.data_padding) || i_layout.batch() == 1) {
                r_node.can_be_optimized(true);
                r_node.requires_reinterpret(true);

                std::vector<int32_t> pad_lo(o_layout.data_padding._lower_size.begin(),
                                            o_layout.data_padding._lower_size.begin() + o_layout.get_rank());
                std::vector<int32_t> pad_hi(o_layout.data_padding._upper_size.begin(),
                                            o_layout.data_padding._upper_size.begin() + o_layout.get_rank());

                pad_lo[0] = i_layout.data_padding._lower_size[0];
                pad_hi[0] = i_layout.data_padding._upper_size[0];

                pad_lo[1] = i_layout.data_padding._lower_size[1];
                pad_hi[1] = i_layout.data_padding._upper_size[1];

                if (i_layout.feature() % 16 != 0) {
                    pad_hi[1] += 16 - i_layout.feature() % 16;
                }

                r_node.merge_output_padding(padding{pad_lo, pad_hi});
                continue;
            }
        }

        if (!o_layout.compatible(i_layout))
            continue;

        if (r_node.is_output() && i_layout.get_linear_size() != o_layout.get_linear_size())
            continue;

        // mark as optimized
        r_node.can_be_optimized(true);
        r_node.requires_reinterpret(!o_layout.identical(i_layout));
        if (o_layout.identical(i_layout)) {  // no need of reshape
            if (r_node.is_output()) {
                // if removed reorder is output, we need to add it's dependency id to the optimized primitives list,
                // because it's name will be changed after extract_and_remove call
                p.add_optimized_primitive_info(r_node.get_dependency(0).get_primitive()->id, {r_node.get_primitive()->id});
            } else {
                p.add_optimized_primitive_info(r_node.get_primitive()->id);
            }

            LOG_NODE_REMOVAL(r_node.id());
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

        if (node->has_fused_primitives())
            continue;

        auto& dep = node->get_dependency(0);

        for (auto& user : dep.get_users()) {
            if (user->is_type<reorder>() &&
                user != node &&
                !user->is_output() &&
                !user->has_fused_primitives()) {
                auto l1 = node->get_output_layout();
                auto l2 = user->get_output_layout();
                // in multiple outputs, remove redundant reorder is only allowed for same output port idx
                auto l1_port_idx = node->get_dependency_with_port(0).second;
                auto l2_port_idx = user->get_dependency_with_port(0).second;

                if (l1.identical(l2) && (l1_port_idx == l2_port_idx))
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

            LOG_NODE_REMOVAL(remove_reorder_node->id());
            p.replace_all_usages(*remove_reorder_node, *node);
            p.add_optimized_primitive_info(remove_reorder_node->id());
            p.remove_all_connections(*remove_reorder_node);
            p.remove_if_dangling(*remove_reorder_node);
        }
    }

    // This pass removed reorder if previous node can store directly to required layout
    itr = p.get_processing_order().begin();
    if (enable_reorder_fusing) {
        while (itr != p.get_processing_order().end()) {
            auto& node_ptr = *itr++;
            if (!node_ptr->is_type<reorder>())  // only care for reorders
                continue;

            auto& node = node_ptr->as<reorder>();

            auto& input = node.input();
            auto output_layout = node.get_output_layout();

            if (!node.is_simple_reorder())
                continue;

            if (input.get_users().size() != 1)
                continue;

            bool same_data_type = input.get_output_layout().data_type == output_layout.data_type;
            bool allowed_dt_conversion_fuse =
                (input.is_type<one_hot>() || input.is_type<permute>() || input.is_type<mvn>() ||
                 input.is_type<concatenation>() || input.is_type<depth_to_space>() || input.is_type<region_yolo>() ||
                 input.is_type<detection_output>() || input.is_type<gather>() || input.is_type<broadcast>() ||
                 input.is_type<select>() || input.is_type<eltwise>()) && !input.is_constant();
            if (!same_data_type && !allowed_dt_conversion_fuse)
                continue;

            if (!lo.can_fuse_reorder_to_prev(input, node, input.get_output_layout().format, output_layout.format))
                continue;

            // Do not opt out result reorder of Loop body network
            bool is_loop_body_network_output = (node.get_program().is_body_program() && node.is_output());
            if (is_loop_body_network_output)
                continue;

            auto old_output_layout_of_input = input.get_output_layout();
            input.set_output_layout(output_layout, false);
            if (input.type()->has_impl_for(input)) {
                // Add fused_primitive_desc of reorder to the previous node which propagates original output layout
                // during shape inference
                if (input.is_type<mvn>() || input.is_type<concatenation>() || input.is_type<gather>() ||
                    input.is_type<broadcast>() || input.is_type<select>() || input.is_type<eltwise>()) {
                    fused_primitive_desc local_desc(node.get_primitive());
                    local_desc.f_param = node.get_fuse_params();
                    local_desc.total_num_deps = node.get_dependencies().size();
                    local_desc.input_layout = old_output_layout_of_input;
                    local_desc.output_layout = output_layout;
                    input.add_fused_primitive(local_desc);
                }

                node.can_be_optimized(true);
                p.add_optimized_primitive_info(node.id());

                LOG_NODE_REMOVAL(node.id());
                p.extract_and_remove(node);
            } else {
                input.set_output_layout(old_output_layout_of_input, false);
            }
        }
    }
    // This pass removed reorder if the next node supports reorder's input format and data type doesn't change
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node_ptr = *itr++;
        if (!node_ptr->is_type<reorder>() || !node_ptr->is_in_data_flow() || node_ptr->get_users().size() != 1 ||
            node_ptr->get_dependencies().size() != 1 || node_ptr->is_dynamic())
            continue;

        auto& node = node_ptr->as<reorder>();
        auto prim_desc = node.get_primitive();

        auto& usr = node_ptr->get_users().front();
        auto& dep = node_ptr->get_dependency(0);

        auto quantize_opt = usr->is_type<quantize>() &&
                            (dep.get_output_layout().format == format::b_fs_yx_fsv16 ||
                             dep.get_output_layout().format == format::bfyx ||
                             (dep.get_output_layout().format == format::fs_b_yx_fsv32 &&
                             !lo.has_all_enabled_onednn_impls_optimization_attribute()));

        auto convert_color_opt = usr->is_type<convert_color>() && prim_desc->has_surface_input();

        if (!quantize_opt && !convert_color_opt)
            continue;

        auto same_data_type = node.get_input_layout().data_type == node.get_output_layout().data_type;
        if (!same_data_type && !convert_color_opt)
            continue;

        dep.merge_output_padding(node.get_output_layout().data_padding);

        LOG_NODE_REMOVAL(node.id());
        p.replace_all_usages(node, dep);
        p.add_optimized_primitive_info(node.id());
        p.remove_all_connections(node);
        p.remove_if_dangling(node);
    }

    // Remove reorder for cldnn convolution bfyx -> fs_b_yx_fsv32. (no case for onednn)
    auto try_fuse_reorder_bfyx_to_fsv32 = [&](reorder_node* node) -> bool {
        if (node->get_users().size() != 1)
            return false;

        auto& usr = node->get_users().front();
        auto& dep = node->get_dependency(0);
        auto  dep_layout = dep.get_output_layout();

        if (!(usr->is_type<convolution>()) ||
            node->get_output_layout().data_type != dep_layout.data_type ||
            dep_layout.format != format::bfyx)
            return false;
        if (usr->as<convolution>().get_preferred_impl_type() == impl_types::onednn)
            return false;
        if (usr->get_output_layout().format != format::fs_b_yx_fsv32)
            return false;

        if (dep.is_type<input_layout>())
            return false;

        // Skip reorder padding fusing when any one of sibling nodes is optimized out or doesn't support padding.
        if (node->get_output_layout().data_padding) {
            if (update_implementations)
                return false;

            for (auto user : dep.get_users()) {
                if (user != node) {
                    if (user->can_be_optimized())
                        return false;

                    auto node_format = node->get_output_layout().format;
                    auto sizes_in_format = layout::format_sizes(node->get_input_layout(0).data_padding._lower_size, node_format);
                    for (size_t axis = 0; axis < sizes_in_format.size(); axis++) {
                        if (!user->is_padding_supported(static_cast<int>(axis),
                            sizes_in_format[axis]))
                            return false;
                    }
                }
            }
        }

        if (usr->as<convolution>().get_primitive()->groups != 1)
            return false;

        dep.merge_output_padding(node->get_output_layout().data_padding);
        LOG_NODE_REMOVAL(node->id());
        p.replace_all_usages(*node, dep);
        p.get_processing_order().erase(node);
        p.add_optimized_primitive_info(node->id());
        p.remove_all_connections(*node);
        p.remove_if_dangling(*node);
        return true;
    };

    // Remove reorder for Convolution b_fs_yx_fsv16 -> bfyx
    auto try_fuse_reorder_fsv16_to_bfyx = [&](reorder_node* node) -> bool {
        auto& input = node->input();

        if (!(input.is_type<convolution>()) ||
            (input.is_dynamic()) ||
            !(input.get_output_layout().format == format::b_fs_yx_fsv16) ||
            !(node->get_output_layout().format == format::bfyx))
            return false;

        if (input.as<convolution>().get_primitive()->groups != 1)
            return false;

        // Avoid onednn convolution selects ref kernel for fsv16 -> bfyx
        if (input.as<convolution>().get_preferred_impl_type() == impl_types::onednn)
            return false;

        if (input.get_users().size() != 1)
            return false;

        auto& input_dep = input.get_dependency(0);
        if (input_dep.get_output_layout().format != format::b_fs_yx_fsv16 ||
            input_dep.get_output_layout().data_type == data_types::u8 ||
            input_dep.get_output_layout().data_type == data_types::i8)
            return false;

        for (auto& user : node->get_users()) {
            // if concat is reorder's user and concat's axis is 0(Batch) or 1(Feature), conv's output would have padding.
            // This padding might lead not to select the optimized conv kernel("convolution_gpu_bfyx_f16")
            if (user->is_type<concatenation>()) {
                auto& concat_node = user->as<concatenation>();
                auto concat_axis = concat_node.get_primitive()->axis;
                if (concat_axis == 0 || concat_axis == 1)
                    return false;
            }
        }

        auto old_output_layout_of_input = input.get_output_layout();
        auto output_layout = node->get_output_layout();
        input.set_output_layout(output_layout, false);
        if (input.type()->has_impl_for(input)) {
            input.set_output_padding(node->get_output_layout().data_padding);

            // Add fused_primitive_desc of reorder to convolution which propagate original output layout to jitter
            fused_primitive_desc local_desc(node->get_primitive());
            local_desc.input_layout = input.get_input_layout(0);  // original convolution's output layout
            node->set_input_layout(local_desc.input_layout);
            local_desc.f_param = node->get_fuse_params();
            local_desc.outer_dep_start_idx = -1;
            local_desc.output_layout = output_layout;
            input.add_fused_primitive(local_desc);

            // remove reorder node
            LOG_NODE_REMOVAL(node->id());
            node->can_be_optimized(true);
            p.add_optimized_primitive_info(node->id());
            p.extract_and_remove(*node);
            return true;
        } else {
            input.set_output_layout(old_output_layout_of_input, false);
            return false;
        }
    };

    if (enable_reorder_fusing) {
        itr = p.get_processing_order().begin();
        while (itr != p.get_processing_order().end()) {
            auto& node = *itr++;
            if (!node->is_type<reorder>())
                continue;

            auto& r_node = node->as<reorder>();

            if (!r_node.is_in_data_flow() || r_node.get_dependencies().size() != 1)
                continue;

            if (!r_node.is_simple_reorder())
                continue;

            // Remove reorder for Convolution bfyx -> fs_b_yx_fsv32
            // Process remaining patterns here that are not removed at the first while loop
            // e.g., reorder with otuput padding
            if (try_fuse_reorder_bfyx_to_fsv32(&r_node))
                continue;
            // Remove reorder for Convolution b_fs_yx_fsv16 -> bfyx
            if (try_fuse_reorder_fsv16_to_bfyx(&r_node))
                continue;
        }
    }

    // Additional reshape chains shrink.
    // This step is needed to handle the cases when the plugin creates patterns like reshape -> reorder -> reshape
    // So these reshapes are not optimized in handle_reshape pass due to reorder between them,
    // but the reorder can be removed by one of the steps above, so we can optimize reshapes after that.
    // In addition this pass can completely remove useless reshapes sequence where the output size is equal to input.
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node = *itr++;
        if (!node->is_type<reshape>())
            continue;

        auto& reshape_node = node->as<reshape>();
        auto& dep_node = reshape_node.get_dependency(0);

        if (!dep_node.is_type<reshape>())
            continue;

        auto& reshape_input_node = dep_node.as<reshape>();

        // In case of new shape infer we should not shrink reshapes chain if first reshape changes input rank, e.g.
        // [a, b] -> reshape1 -> [a1, b1, c1] -> reshape2 -> [a2, b2, 0] and any of the reshapes has special_zero=true
        // Configuration above will fail if we remove reshape1 node as attempt to handle special zero will fail due to small rank of input
        if (p.is_new_shape_infer() &&
            reshape_node.get_output_pshape().size() != dep_node.get_input_pshape().size() &&
            (reshape_node.get_primitive()->special_zero || reshape_input_node.get_primitive()->special_zero))
            continue;

        if (reshape_node.is_dynamic())
            continue;

        bool remove_dep = reshape_input_node.get_users().size() == 1 && !reshape_input_node.is_output() &&
                          !reshape_input_node.has_fused_primitives();
        bool remove_current = remove_dep && !reshape_input_node.get_dependencies().empty() &&
                              reshape_input_node.get_input_layout(0) == reshape_node.get_output_layout() &&
                              !reshape_node.has_fused_primitives();

        if (remove_dep) {
            LOG_NODE_REMOVAL(reshape_input_node.id());
            reshape_input_node.can_be_optimized(true);
            p.add_optimized_primitive_info(reshape_input_node.id());
            p.extract_and_remove(reshape_input_node);
        }

        if (remove_current) {
            LOG_NODE_REMOVAL(reshape_node.id());
            reshape_node.can_be_optimized(true);
            p.add_optimized_primitive_info(reshape_node.id());
            p.extract_and_remove(reshape_node);
        }
    }

    // Remove reorders before shape_of primitive
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<reorder>() || node->has_fused_primitives() ||
            !node->is_in_data_flow() || node->get_users().size() != 1 ||
            !node->get_users().front()->is_type<shape_of>())
            continue;

        auto& dep = node->get_dependency(0);

        LOG_NODE_REMOVAL(node->id());
        p.replace_all_usages(*node, dep);
        p.add_optimized_primitive_info(node->id());
        p.remove_all_connections(*node);
        p.remove_if_dangling(*node);
    }

    for (auto n : p.get_processing_order()) {
        if (n->is_in_data_flow() && n->is_type<reorder>()) {
            auto preferred_impl = lo.get_preferred_impl_type(*n, n->get_input_layout(0).format);
            n->set_preferred_impl_type(preferred_impl);
        }
    }

    // Recalculate processing order if it is not correct
    bool is_correct = true;
    for (auto node : p.get_processing_order()) {
        if (!p.get_processing_order().is_correct(node)) {
            is_correct = false;
            break;
        }
    }

    if (!is_correct) {
        p.get_processing_order().calc_processing_order(p);
    }
}
