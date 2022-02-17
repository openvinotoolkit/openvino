// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "intel_gpu/graph/program.hpp"
#include "program_helpers.h"
#include "runtime/cldnn_itt.hpp"
#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>

using namespace cldnn;

void basic_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::BasicMemoryDependencies");
    auto itr = p.get_processing_order().begin();
    std::vector<primitive_id> past_outputs;
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies()) {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        if (node->is_type<convolution>() && node->get_preferred_impl_type() == impl_types::onednn) {
            auto& conv = node->as<convolution>();
            bool can_reuse_eltwise_mem = false;
            size_t eltw_dep = 0;

            for (auto& fused_op : conv.get_fused_primitives()) {
                if (fused_op.node->is_type<eltwise>() && fused_op.deps.size() == 1) {
                    auto eltw_in_layout = conv.get_dependency(fused_op.dep_start_idx).get_output_layout();
                    auto conv_out_layout = node->get_output_layout();
                    if (eltw_dep > 0) {
                        can_reuse_eltwise_mem = false;
                        break;
                    }

                    if (eltw_in_layout.size == conv_out_layout.size &&
                        eltw_in_layout.format == conv_out_layout.format &&
                        eltw_in_layout.data_padding == conv_out_layout.data_padding &&
                        data_type_traits::size_of(eltw_in_layout.data_type) == data_type_traits::size_of(conv_out_layout.data_type)) {
                        eltw_dep = fused_op.dep_start_idx;
                        can_reuse_eltwise_mem = true;
                    }
                }
            }

            if (can_reuse_eltwise_mem) {
                auto& eltw_node = conv.get_dependency(eltw_dep);
                eltw_node.can_share_buffer(false);
                conv.can_share_buffer(false);
                for (auto& user : conv.get_users()) {
                    add_memory_dependency(user, &eltw_node);
                    add_memory_dependency(user, &conv);
                }
            }
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}
