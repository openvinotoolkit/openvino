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
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <set>
#include <vector>

using namespace cldnn;

// F5 Debug assertion D2 (zero Release cost). Temporarily define F5_DEBUG_CHECKS to 1 to exercise it in an NDEBUG build.
#ifndef F5_DEBUG_CHECKS
#    ifndef NDEBUG
#        define F5_DEBUG_CHECKS 1
#    endif
#endif

// Phase 0.5a B4 Debug assertion (zero Release cost). Temporarily define ALIAS_B4_DEBUG_CHECKS to 1 to exercise it in an NDEBUG build.
#ifndef ALIAS_B4_DEBUG_CHECKS
#    ifndef NDEBUG
#        define ALIAS_B4_DEBUG_CHECKS 1
#    endif
#endif

void basic_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::BasicMemoryDependencies");
    auto itr = p.get_processing_order().begin();
    std::vector<size_t> past_outputs;
    size_t owner_diff_count = 0;
#ifdef ALIAS_B4_DEBUG_CHECKS
    // Defect B transition (Phase 0.5a): nodes marked only by the new owner resolution, with their can_share_buffer before the mark.
    struct owner_only_mark {
        program_node* node;
        bool was_non_shareable;
    };
    std::vector<owner_only_mark> owner_only_marks;
    std::set<const program_node*> legacy_marked;  // nodes the legacy sum walk marked (anywhere in the pass)
#endif
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
            // As for the oneDNN sum post-op: keep both ends of the alias out of pool sharing.
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
            // F5: runs after the single re-freeze point, so the answer is "frozen AND live" (M8 in get_reused_eltwmem_idx) and agrees
            // with the dst bindings of the network. Only the in-place sum is handled here; an in-place mul is bound as a restriction
            // carry-over in pass_manager.h (add_memory_dependency chain) and is not a pool-exempt writer.
            const int32_t reused_idx = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(*node);
            bool reused_is_sum = false;
            if (reused_idx != -1) {
                for (const auto& fused_op : node->get_fused_primitives()) {
                    if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1 && fused_op.has_outer_dep() &&
                        fused_op.outer_dep_start_idx == reused_idx && fused_op.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
                        reused_is_sum = true;
                        break;
                    }
                }
            }
            if (reused_is_sum) {
                auto& eltw_node = node->get_dependency(static_cast<size_t>(reused_idx));
                node->can_share_buffer(false);
                // Walk up the in-place chain to mark the root buffer owner as non-shareable.
                auto* root = &eltw_node;
                while (root->can_be_optimized() && !root->get_dependencies().empty()) {
                    root = &root->get_dependency(0);
                }
                root->can_share_buffer(false);
#ifdef ALIAS_B4_DEBUG_CHECKS
                legacy_marked.insert(root);
#endif

                // Defect B (concat in the chain): the walk above steps to dep(0) of an in-place concat, i.e. to its first predecessor
                // instead of the holder. The owner by edge direction (slot step first) is the real allocator; mark it as well. The old
                // marking is kept so that no node becomes poolable (neutral transition).
                // SUNSET: P1-C deletes the old walk above and keeps only this owner-based marking.
                program_node* owner = const_cast<program_node*>(
                    program_helpers::resolve_buffer_owner(eltw_node,
                                                         program_helpers::view_resolution::settled_only,
                                                         /*collect_path=*/false,
                                                         /*start_port=*/node->get_dependency_with_port(static_cast<size_t>(reused_idx)).second)
                        .owner);
                if (owner != root) {
                    ++owner_diff_count;
#ifdef ALIAS_B4_DEBUG_CHECKS
                    owner_only_marks.push_back({owner, !owner->can_share_buffer()});
#endif
                    owner->can_share_buffer(false);
                }

                for (const auto& user : node->get_users()) {
                    add_memory_dependency(user, &eltw_node);
                    add_memory_dependency(user, node);
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
                for (const auto& dep : node->get_dependencies()) {
                    dep.first->set_output(true);
                }
            }
        }
    }

    if (owner_diff_count != 0) {
        GPU_DEBUG_LOG << "[alias_analysis] defect B: sum owner differs from the legacy chain root " << owner_diff_count << " time(s)" << std::endl;
    }

#ifdef ALIAS_B4_DEBUG_CHECKS
    // B4: judged after the main loop (mutable_data propagates the output flag inside it). A node marked only by the owner resolution
    // must already be outside the pool, otherwise this transition would have changed an allocation.
    for (const auto& m : owner_only_marks) {
        const program_node& n = *m.node;
        if (legacy_marked.count(&n)) {
            continue;  // also marked by the legacy walk of another sum: not owner-only
        }
        // design 11 section 7.3: a static cbo node that is not runtime skippable bypasses the pool for certain
        OPENVINO_ASSERT(m.was_non_shareable || (n.can_be_optimized() && !n.is_dynamic() && !n.is_runtime_skippable()) || n.is_output() || p.is_internal_program(),
                        "[GPU] alias_analysis B4: node ", n.id(), " is newly marked non-shareable by the defect B owner resolution but was poolable");
    }
#endif  // ALIAS_B4_DEBUG_CHECKS

#if defined(F5_DEBUG_CHECKS) && defined(ENABLE_ONEDNN_FOR_GPU)
    // D2: every static node still carrying a frozen sum (F5 left only consistent ones) is the pool-exempt writer this pass just
    // marked, and get_reused_eltwmem_idx names the same dependency.
    for (const auto& node : p.get_processing_order()) {
        if (node->is_dynamic()) {
            continue;
        }
        for (const auto& e : node->get_fused_primitives_onednn()) {
            if (e.op_type != onednn_post_op_type::sum && e.op_type != onednn_post_op_type::optimized_sum) {
                continue;
            }
            OPENVINO_ASSERT(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(*node) == static_cast<int32_t>(e.mem_dep) &&
                                !node->can_share_buffer(),
                            "[GPU] F5 D2: node ", node->id(), " has a frozen sum on dep ", e.mem_dep,
                            " but basic_memory_dependencies did not mark it as the in-place writer");
        }
    }
#endif  // F5_DEBUG_CHECKS
}
