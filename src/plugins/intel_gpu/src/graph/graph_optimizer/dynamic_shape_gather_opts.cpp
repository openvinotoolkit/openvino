// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "gather_inst.h"
#include "program_helpers.h"

using namespace cldnn;

void dynamic_shape_gather_opts::run(program& p) {
    auto itr = p.get_processing_order().begin();
    // Set gathers that might be skipped at runtime as can_be_optimized.
    // If not set, memory dependency will not work for the nodes that are skipped at runtime
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<gather>())
            continue;
        auto& gather_node = node->as<gather>();
        // Check pattern
        auto impl_params = gather_node.get_kernel_impl_params();
        if (gather_node.has_fused_primitives() ||
            (impl_params->get_input_layout(0).data_type != impl_params->get_output_layout().data_type) ||
            gather_node.get_dependency(1).is_constant() || gather_node.get_dependency(1).is_type<data>())
            continue;
        auto idx_rank = impl_params->get_input_layout(1).get_partial_shape().size();

        if (idx_rank != 1) {
            continue;
        }
        auto axis = impl_params->typed_desc<gather>()->axis;
        if (impl_params->get_input_layout(0).get_partial_shape()[axis] == -1
            || impl_params->get_input_layout(1).get_partial_shape()[0] == -1
            || impl_params->get_input_layout(0).get_partial_shape()[axis] == impl_params->get_input_layout(1).get_partial_shape()[0]) {
            // May be skipepd
            gather_node.can_be_optimized(true);
            GPU_DEBUG_TRACE_DETAIL << "[dynamic_shape_gather_opts] : " << gather_node.id() << "can_be_optimized" << std::endl;
        }
    }
}
