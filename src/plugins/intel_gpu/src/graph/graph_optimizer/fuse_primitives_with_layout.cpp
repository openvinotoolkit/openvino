// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_helpers.h"
#include "pass_manager.h"

#include "eltwise_inst.h"
#include "quantize_inst.h"

#include <vector>

using namespace cldnn;

void fuse_primitives_with_layout::run(program& p) {
    auto eltwise_supports_fusings = [&](eltwise_node& node) -> bool {
        auto out_layout = node.get_output_layout();
        // This condition refers to optimizied kernel EltwiseKernel_fs_b_yx_fsv32
        if (out_layout.data_type == data_types::f16 && out_layout.batch() > 1 &&
            (p.get_layout_optimizer().get_optimization_attributes().fs_b_yx_fsv32_network || out_layout.format == format::fs_b_yx_fsv32)) {
            return false;
        }

        return true;
    };

    bool need_recalc_processing_order = false;
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;

    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output() || node->is_constant())
            continue;

        // No optimized Eltwise kernel supports fused-operation for fs_b_yx_fsv32
        // Check fusing quantize to eltwise for this case
        auto func_fuse_quantize = [&](quantize_node& node) {
            bool should_fuse = false;
            auto out_layout = node.get_output_layout();
            if (out_layout.is_dynamic() || node.is_in_shape_of_subgraph())
                return;

            auto& input_node = node.get_dependency(0);
            auto in_layout = input_node.get_output_layout();
            if (input_node.get_users().size() != 1 || input_node.get_dependencies().empty() ||
                in_layout.is_dynamic() || input_node.is_in_shape_of_subgraph() || in_layout.format != out_layout.format)
                return;

            should_fuse |= input_node.is_type<eltwise>() && eltwise_supports_fusings(input_node.as<eltwise>());
            if (!should_fuse)
                return;

            p.fuse_nodes(input_node, node, &fusing_history);
            need_recalc_processing_order = true;
        };

        program_helpers::do_for_types<quantize>(*node, func_fuse_quantize);
    }

    // Need to update processing order when peer node processing number is greater than fused node
    if (need_recalc_processing_order)
        p.get_processing_order().calc_processing_order(p);
}
