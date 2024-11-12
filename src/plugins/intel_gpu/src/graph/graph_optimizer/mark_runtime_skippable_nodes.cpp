// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "gather_inst.h"
#include "non_max_suppression_inst.h"
#include "permute_inst.h"
#include "strided_slice_inst.h"
#include "kv_cache_inst.h"
#include "gemm_inst.h"
#include "shape_of_inst.h"
#include "broadcast_inst.h"
#include "non_zero_inst.h"
#include "non_max_suppression_inst.h"
#include "unique_inst.hpp"
#include "scatter_elements_update_inst.h"
#include "scatter_update_inst.h"
#include "scatter_nd_update_inst.h"
#include "program_helpers.h"

using namespace cldnn;

void mark_runtime_skippable_nodes::run(program& p) {
    auto itr = p.get_processing_order().begin();

    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        // Set gathers that might be skipped at runtime as can_be_optimized.
        // If not set, memory dependency will not work for the nodes that are skipped at runtime
        if (node->is_type<data>() || node->is_constant())
            continue;

        std::function<bool(const program_node& node)> all_users_are_shape_of = [&](const program_node& node) {
            if (node.is_input() || node.is_output())
                return false;
            for (auto& u : node.get_users()) {
                if (!u->is_type<shape_of>())
                    return false;
            }
            return true;
        };

        if (all_users_are_shape_of(*node) &&
            // primitives that should be executed to know output shapes
            !node->is_type<gather_nonzero>() && !node->is_type<unique_gather>() &&
            !node->is_type<non_max_suppression_gather>()) {
            // always to skip, no runtime execution
            node->can_be_optimized(true);
            GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node->id() << " has only shape_of as users. Set can_be_optimized always"
                                   << std::endl;
            continue;
        }

        program_helpers::do_for_types<gather>(*node, [](gather_node& node) {
            // Check pattern
            auto impl_params = node.get_kernel_impl_params();
            if (node.has_fused_primitives() ||
                (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type) ||
                node.get_dependency(1).is_constant() || node.get_dependency(1).is_type<data>())
                return;
            auto idx_rank = impl_params->get_input_layout(1).get_partial_shape().size();

            if (idx_rank != 1) {
                return;
            }
            auto axis = impl_params->typed_desc<gather>()->axis;
            if (impl_params->get_input_layout(0).get_partial_shape()[axis] == -1
                || impl_params->get_input_layout(1).get_partial_shape()[0] == -1
                || impl_params->get_input_layout(0).get_partial_shape()[axis] == impl_params->get_input_layout(1).get_partial_shape()[0]) {
                // May be skipped
                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });
        program_helpers::do_for_types<permute>(*node, [](permute_node& node){
            // if node is already optimized at compilation time, do not handle at runtime
            if (node.can_be_optimized())
                return;
            auto impl_params = node.get_kernel_impl_params();
            if (node.is_output() ||
                node.has_fused_primitives() ||
                (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type) ||
                impl_params->get_input_layout(0).data_padding.is_dynamic())
                return;

            // TODO: For now, all permutes with dynamic shape are applied.
            //       A more detailed pattern will need to be applied later
            if (node.is_dynamic()) {
                if (node.get_dependency(0).is_type<kv_cache>())
                    return;
                // If the user is concatenation, priority should be given to in place concat optimization at runtime
                if (node.have_user_with_type<concatenation>() && node.get_users().size() == 1)
                    return;
                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });
        program_helpers::do_for_types<strided_slice>(*node, [](strided_slice_node& node){
            auto impl_params = node.get_kernel_impl_params();
            if (node.is_output()
                || node.has_fused_primitives()
                || (impl_params->get_input_layout(0).format != impl_params->get_output_layout().format)
                || (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type))
                return;

            auto prim = impl_params->typed_desc<strided_slice>();
            auto begin = prim->begin;
            auto strides = prim->strides;
            auto begin_mask = prim->begin_mask;
            if (prim->end_mask.empty()
                || !prim->new_axis_mask.empty()
                || !prim->shrink_axis_mask.empty()
                || !prim->ellipsis_mask.empty()
                || !(all_zeroes(begin) || all_ones(begin_mask))
                || !all_ones(strides))
                return;

            auto end = prim->end;
            auto end_mask = prim->end_mask;
            auto in_ps = impl_params->get_input_layout(0).get_partial_shape();
            bool is_valid = false;
            bool is_equal_size = (end.size() == end_mask.size());
            for (size_t i = 0; i < end.size(); i++) {
                if ((is_equal_size && end_mask[i] == 1) || (in_ps[i].is_static() && end[i] == in_ps[i].get_length())) {
                    is_valid = true;
                } else {
                    is_valid = false;
                }
            }
            if (!end.empty() && !is_valid)
                return;
            node.can_be_optimized(true);
            // Set runtime skippable only when the node is set as can_be_optimized finally.
            node.set_runtime_skippable(true);
            GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
        });
        program_helpers::do_for_types<broadcast>(*node, [](broadcast_node& node){
            auto impl_params = node.get_kernel_impl_params();
            if (node.is_output()
                || node.has_fused_primitives()
                || (impl_params->get_input_layout(0).format != impl_params->get_output_layout().format)
                || (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type))
                return;

            if (node.is_dynamic()) {
                // If the user is reorder, it could be fused to broadcast in the remove_redundant_reorders pass.
                // In this case, broadcast can not be optimized due to different input and output shapes.
                if (node.have_user_with_type<reorder>() && node.get_users().size() == 1)
                    return;

                // Check if the size of rank is different, or if one of static dimensions has different size
                auto input_pshape = impl_params->get_input_layout(0).get_partial_shape();
                auto output_pshape = impl_params->get_output_layout().get_partial_shape();

                if (input_pshape.rank().is_static() && output_pshape.rank().is_static()) {
                    if (input_pshape.size() != output_pshape.size())
                        return;

                    auto input_pdim = input_pshape.begin();
                    auto output_pdim = output_pshape.begin();
                    while (input_pdim != input_pshape.end()) {
                        if (input_pdim->is_static() && output_pdim->is_static()) {
                            if (input_pdim->get_max_length() != output_pdim->get_max_length())
                                return;
                        }

                        input_pdim++;
                        output_pdim++;
                    }
                }

                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });
        program_helpers::do_for_types<reorder>(*node, [](reorder_node& node){
            auto impl_params = node.get_kernel_impl_params();

            if ((node.is_output() && node.get_dependency(0).is_input())
                || node.has_fused_primitives()
                || (impl_params->get_input_layout(0).format != impl_params->get_output_layout().format)
                || (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type)
                || (impl_params->get_input_layout(0).data_padding != impl_params->get_output_layout().data_padding))
                return;

            if (node.is_dynamic()) {
                if (!node.is_output() && node.get_users().size() != 1)
                    return;

                // If the user is concatenation with 1 user and the concat is optimized, priority should be given to in place concat optimization at runtime
                if (node.have_user_with_type<concatenation>() && node.get_users().front()->can_be_optimized())
                    return;

                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });

        program_helpers::do_for_types<scatter_elements_update>(*node, [](scatter_elements_update_node & node){
            auto impl_params = node.get_kernel_impl_params();

            if ((node.is_output() && node.get_dependency(0).is_input())
                || node.has_fused_primitives()
                || (impl_params->get_input_layout(0).format != impl_params->get_output_layout().format)
                || (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type))
                return;

            if (node.is_dynamic()) {
                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });

        program_helpers::do_for_types<scatter_update>(*node, [](scatter_update_node & node){
            auto impl_params = node.get_kernel_impl_params();

            if ((node.is_output() && node.get_dependency(0).is_input())
                || node.has_fused_primitives()
                || (impl_params->get_input_layout(0).format != impl_params->get_output_layout().format)
                || (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type))
                return;

            if (node.is_dynamic()) {
                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });

        program_helpers::do_for_types<scatter_nd_update>(*node, [](scatter_nd_update_node & node){
            auto impl_params = node.get_kernel_impl_params();

            if ((node.is_output() && node.get_dependency(0).is_input())
                || node.has_fused_primitives()
                || (impl_params->get_input_layout(0).format != impl_params->get_output_layout().format)
                || (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type))
                return;

            if (node.is_dynamic()) {
                node.can_be_optimized(true);
                // Set runtime skippable only when the node is set as can_be_optimized finally.
                node.set_runtime_skippable(true);
                GPU_DEBUG_TRACE_DETAIL << "[mark_runtime_skippable_nodes] : " << node.id() << " can_be_optimized" << std::endl;
            }
        });
    }
}
