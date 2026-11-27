// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "program_helpers.h"
#include "stateless_kv_inst.h"
#include "intel_gpu/runtime/itt.hpp"
#include <algorithm>
#include <vector>

using namespace cldnn;

namespace {
// Nodes whose buffer the output may expose as its own at runtime: an optimized-out output is a view of the buffer of
// its input, and a condition may pass the buffer of an input through (see condition_node::get_inputs_forwarded_to_outputs()).
// Unlike the buffer of the output itself, those buffers may come from the memory pool.
std::vector<program_node*> get_output_aliases(program_node& output) {
    std::vector<program_node*> aliases;
    auto* node = &output;
    while (true) {
        if (node->is_type<condition>()) {
            const auto forwarded = node->as<condition>().get_inputs_forwarded_to_outputs();
            aliases.insert(aliases.end(), forwarded.begin(), forwarded.end());
        }
        if ((!node->can_be_optimized() && !node->is_runtime_skippable()) || node->get_dependencies().empty()) {
            break;
        }
        node = &node->get_dependency(0);
        aliases.push_back(node);
    }
    return aliases;
}
}  // namespace

void basic_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::BasicMemoryDependencies");
    auto itr = p.get_processing_order().begin();
    std::vector<size_t> past_outputs;
    // Nodes whose buffer one of past_outputs may expose, see get_output_aliases()
    std::vector<program_node*> past_output_aliases;
    while (itr != p.get_processing_order().end()) {
        const auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>()) {
            continue;
        }

        // stateless_kv may update input 0 (`past`) in place.
        const bool inplace_kv = node->is_type<stateless_kv>();

        // add my dependencies to restriction list (can't share input.output buffers)
        const auto& deps = node->get_dependencies();
        for (size_t i = 0; i < deps.size(); ++i) {
            if (inplace_kv && i == 0) {
                continue;
            }
            add_memory_dependency(node, deps[i].first);
            add_memory_dependency(deps[i].first, node);
        }

        if (inplace_kv) {
            // The KV cache is written in place into its past buffer: keep both ends of the alias out of pool sharing.
            node->can_share_buffer(false);
            auto* root = &node->get_dependency(0);
            while (root->can_be_optimized() && !root->get_dependencies().empty()) {
                root = &root->get_dependency(0);
            }
            root->can_share_buffer(false);
        }

        // LoRA can reuse the memory of the previous node, but not be optimized
        // So memory dependencies need to be expanded with LoRA dependencies
        if (node->have_user_with_type<lora>()) {
            for (const auto& it : node->get_users().front()->get_dependencies()) {
                add_memory_dependency(node, it.first);
                add_memory_dependency(it.first, node);
            }
        }

        if (node->get_preferred_impl_type() == impl_types::onednn) {
            size_t eltw_dep = 0;
            for (auto& fused_op : node->get_fused_primitives()) {
                if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                    // If it is first sum, reuse the buffer
                    auto fusing_type = onednn_eltwise_fusing_helpers::get_add_fusing_type(*node, fused_op);
                    if (fusing_type != add_fusing_type::sum || eltw_dep != 0) {
                        continue;
                    }
                    if (!fused_op.has_outer_dep()) {
                        continue;
                    }
                    eltw_dep = fused_op.outer_dep_start_idx;
                    auto& eltw_node = node->get_dependency(eltw_dep);
                    node->can_share_buffer(false);
                    // Walk up the in-place chain to mark the root buffer owner as non-shareable. An eltw_node that owns
                    // its buffer stays in the pool: add_memory_dependency extends its restrictions through this node.
                    auto* root = &eltw_node;
                    while (root->can_be_optimized() && !root->get_dependencies().empty()) {
                        root = &root->get_dependency(0);
                    }
                    if (root != &eltw_node) {
                        root->can_share_buffer(false);
                    }

                    for (const auto& user : node->get_users()) {
                        add_memory_dependency(user, &eltw_node);
                        add_memory_dependency(user, node);
                    }
                }
            }
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        // The same holds for the buffers an output may expose. They may come from the pool, which only checks the
        // restrictions of the requesting node, and static memory is allocated in size order, so restrict both ways.
        for (auto* alias : past_output_aliases) {
            add_memory_dependency(node, alias);
            add_memory_dependency(alias, node);
        }
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output()) {
            past_outputs.push_back(node->get_unique_id());
            for (auto* alias : get_output_aliases(*node)) {
                if (std::find(past_output_aliases.begin(), past_output_aliases.end(), alias) == past_output_aliases.end()) {
                    past_output_aliases.push_back(alias);
                }
            }
            if (node->is_type<mutable_data>()) {
                // if output is mutable data, then propagate output flag to its dependencies
                for (const auto& dep : node->get_dependencies()) {
                    dep.first->set_output(true);
                }
            }
        }
    }
}
