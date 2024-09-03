// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "program_helpers.h"
#include "intel_gpu/runtime/itt.hpp"
#include <vector>

using namespace cldnn;

void basic_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::BasicMemoryDependencies");
    auto itr = p.get_processing_order().begin();
    std::vector<size_t> past_outputs;
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (const auto& it : node->get_dependencies()) {
            add_memory_dependency(node, it.first);
            add_memory_dependency(it.first, node);
        }

        if (node->get_preferred_impl_type() == impl_types::onednn) {
            size_t eltw_dep = 0;
            for (auto& fused_op : node->get_fused_primitives()) {
                if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                    // If it is first sum, reuse the buffer
                    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(*node, fused_op);
                    if (fusing_type != add_fusing_type::sum || eltw_dep != 0)
                        continue;
                    if (!fused_op.has_outer_dep())
                        continue;
                    eltw_dep = fused_op.outer_dep_start_idx;
                    auto& eltw_node = node->get_dependency(eltw_dep);
                    eltw_node.can_share_buffer(false);
                    node->can_share_buffer(false);
                    for (auto& user : node->get_users()) {
                        add_memory_dependency(user, &eltw_node);
                        add_memory_dependency(user, node);
                    }
                }
            }
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output()) {
            past_outputs.push_back(node->get_unique_id());
            if (node->is_type<mutable_data>()) {
                // if output is mutable data, then propagate output flag to its dependencies
                for (auto& dep : node->get_dependencies()) {
                    dep.first->set_output(true);
                }
            }
        }
    }
}
