// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "pass_manager.h"
#include "program_node.h"
#include "program_helpers.h"
#include "primitive_type.h"
#include "primitive_inst.h"
#include "alias_analysis.h"
#include "alias_validation.h"
#include "exec_axis.h"
#include "intel_gpu/graph/program.hpp"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <impls/onednn/utils.hpp>
#include "intel_gpu/graph/serialization/weights_reorder_params.hpp"
#endif

using namespace cldnn;

// F5 Debug assertions D1/D2 (zero Release cost). Temporarily define F5_DEBUG_CHECKS to 1 to exercise them in an NDEBUG build.
// Phase 1a Debug assertion I-1a-4 (zero Release cost); D6 lives in program.cpp under the same macro.
#ifndef INPLACE_CONFLICT_DEBUG_CHECKS
#    ifndef NDEBUG
#        define INPLACE_CONFLICT_DEBUG_CHECKS 1
#    endif
#endif

#ifndef F5_DEBUG_CHECKS
#    ifndef NDEBUG
#        define F5_DEBUG_CHECKS 1
#    endif
#endif

void add_onednn_optimization_attributes::run(program& p) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    for (const auto& node : p.get_processing_order()) {
        if (node->get_preferred_impl_type() == impl_types::onednn
            && !node->is_dynamic()) {
            node->init_onednn_primitive_attributes();
        }
    }
#endif // ENABLE_ONEDNN_FOR_GPU
}

#ifdef ENABLE_ONEDNN_FOR_GPU
namespace {
// The weights md the oneDNN impl expects: empty if it needs no weights reorder.
std::shared_ptr<dnnl::memory::desc> expected_weights_md(const primitive_impl& impl) {
    auto params = impl.get_weights_reorder_params();
    if (!params) {
        return nullptr;
    }
    auto onednn_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(params);
    OPENVINO_ASSERT(onednn_params != nullptr, "[GPU] F5: a oneDNN impl carries non-oneDNN weights reorder params");
    return onednn_params->_out_desc;
}

// Re-creates the oneDNN impl of `node` from the patched attributes. `what` only feeds the error messages (e.g. "downgrading the in-place sum
// to binary_add"). Same impl type only (C4); fails loudly when the new impl expects a
// weights layout other than the one the weights dependency was already reordered to (C2: no reorder insertion at this stage, it runs
// after propagate_constants, so a freshly inserted constant reorder would be neither folded nor given an impl).
void rebuild_onednn_impl(program& p, program_node& node, const char* what) {
    const auto* old_impl = node.get_selected_impl();
    OPENVINO_ASSERT(old_impl != nullptr && old_impl->is_onednn());
    const auto old_manager = old_impl->m_manager;
    const auto old_weights_md = expected_weights_md(*old_impl);

    auto params = node.get_kernel_impl_params();
    std::unique_ptr<primitive_impl> new_impl;
    if (old_manager != nullptr) {
        try {
            new_impl = old_manager->create(node, *params);
        } catch (const std::exception& e) {
            GPU_DEBUG_LOG << "[F5] re-creating the impl of " << node.id() << " with its manager failed: " << e.what() << std::endl;
        }
    }
    if (!new_impl) {
        try {
            new_impl = node.type()->create_impl(node);
        } catch (const std::exception& e) {
            OPENVINO_THROW("[GPU] F5: cannot re-create the oneDNN impl of ", node.id(), " after ", what, ": ", e.what());
        }
    }
    OPENVINO_ASSERT(new_impl && new_impl->is_onednn(),
                    "[GPU] F5: re-creating the impl of ", node.id(), " after ", what, " did not give a oneDNN impl");

    if (!p.is_internal_program()) {
        const auto new_weights_md = expected_weights_md(*new_impl);
        const bool same = (!old_weights_md && !new_weights_md) || (old_weights_md && new_weights_md && *old_weights_md == *new_weights_md);
        OPENVINO_ASSERT(same,
                        "[GPU] F5: ", what, " (node ", node.id(), ")",
                        " changes the weights layout the oneDNN impl expects; the weights dependency was reordered for the old impl");
    }

    node.set_selected_impl(std::move(new_impl));
    auto* impl = node.get_selected_impl();
    auto new_params = node.get_kernel_impl_params();
    p.get_kernels_cache().add_kernels_source(*new_params, impl->get_kernels_source());
}

// Phase 1a: a frozen in-place oneDNN post-op of a static node (sum or binary_mul_inplace) that survived the F5 structural downgrade.
struct inplace_candidate {
    program_node* node = nullptr;
    int32_t mem_dep = -1;
    edge_origin origin = edge_origin::onednn_sum;
};

// Builds a plan that downgrades every candidate (no analysis involved): used when the analysis cannot give a verdict (no usable axis, or an
// exception in Release). Downgrading is always the safe direction.
inplace_conflict_plan downgrade_all(const std::vector<inplace_candidate>& cand, war_downgrade_reason why) {
    inplace_conflict_plan plan;
    for (const auto& c : cand) {
        inplace_conflict_action a;
        a.writer = c.node;
        a.mem_dep = c.mem_dep;
        a.origin = c.origin;
        a.why = why;
        plan.actions.push_back(std::move(a));
    }
    return plan;
}

// The switch and the program-level gates. Returns the reason the pass is skipped for the whole program ("" = run), design 17 §5.3.
std::string program_level_skip_reason(const program& p, const std::vector<inplace_candidate>& cand) {
    if (!p.get_config().get_enable_inplace_conflict_resolution()) {
        return "disabled";
    }
    if (cand.empty()) {
        return "no_candidates";
    }
    if (p.is_internal_program()) {
        return "internal";   // internal programs (propagate_constants, unfused subgraphs) have no usable network-level time axis
    }
    if (p.is_body_program()) {
        return "body";       // 1a-BODY: body networks have their own order and back edges
    }
    if (p.get_config().get_queue_type() != QueueTypes::in_order) {
        return "oooq";       // unreachable with oneDNN (execution_config forces in_order), kept as the same guard war_safe has
    }
    return "";
}

}  // namespace
#endif  // ENABLE_ONEDNN_FOR_GPU

void refreeze_onednn_inplace_post_ops::apply_inplace_conflict_plan(program& p, const inplace_conflict_plan& plan, inplace_conflict_report& rep) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::vector<program_node*> rebuild;
    for (const auto& a : plan.actions) {
        if (a.why == war_downgrade_reason::none) {
            ++rep.kept;
            continue;
        }
        auto& n = p.get_node(a.writer->id());
        const bool onednn = n.get_selected_impl() != nullptr && n.get_selected_impl()->is_onednn();
        const bool is_mul = a.origin == edge_origin::onednn_mul;
        // origin onednn_sum (hypothetical edges only exist in tests, where they are never applied) or onednn_mul. A non-oneDNN impl (OCL can
        // carry a frozen mul_inplace, F17 G-impl) only gets its label changed: no md / attrs / impl.
        bool changed = false;
        if (is_mul) {
            changed = onednn ? n.downgrade_onednn_mul_inplace_to_binary_mul(static_cast<size_t>(a.mem_dep))
                             : n.relabel_frozen_mul_inplace_as_binary_mul(static_cast<size_t>(a.mem_dep));
        } else {
            changed = onednn ? n.downgrade_onednn_sum_to_binary_add(static_cast<size_t>(a.mem_dep))
                             : n.relabel_frozen_sum_as_binary_add(static_cast<size_t>(a.mem_dep));
        }
        OPENVINO_DEBUG_ASSERT(changed, "[GPU] 1a: no frozen in-place entry for ", n.id(), " dep ", a.mem_dep);
        GPU_DEBUG_LOG << "[1a] downgrade " << n.id() << " dep " << a.mem_dep << " origin " << to_string(a.origin) << " why " << to_string(a.why)
                      << " reasons " << a.r.reasons << std::endl;
        if (changed) {
            if (onednn && std::find(rebuild.begin(), rebuild.end(), &n) == rebuild.end()) {
                rebuild.push_back(&n);
            }
            inplace_conflict_report::entry e;
            e.writer_id = n.id();
            e.mem_dep = a.mem_dep;
            e.origin = a.origin;
            e.why = a.why;
            e.reasons = a.r.reasons;
            rep.entries.push_back(std::move(e));
            for (uint32_t bit = 1; bit != 0 && bit <= wr_no_other_reader; bit <<= 1) {
                if (a.r.reasons & bit) {
                    ++rep.downgrade_bits[bit];
                }
            }
        }
    }
    rep.via_upstream_after_downgrade = plan.via_upstream_after_downgrade;
    // one rebuild per node (a node holds at most one in-place post-op, design H8, so F5 and 1a never both rebuild the same node)
    for (auto* n : rebuild) {
        rebuild_onednn_impl(p, *n, "downgrading the in-place post-op (sum to binary_add / binary_mul_inplace to binary_mul, Phase 1a)");
    }
#    ifdef INPLACE_CONFLICT_DEBUG_CHECKS
    // I-1a-4: the downgraded nodes are consistent again: compiled chain matches the list, and no sum slot remains at the downgraded dependency.
    for (const auto* n : rebuild) {
        const auto* impl = n->get_selected_impl();
        OPENVINO_ASSERT(impl != nullptr && impl->post_op_structure_matches(get_compiled_post_op_slots(n->get_fused_primitives_onednn(), 0, {}).kinds),
                        "[GPU] 1a I-1a-4: the rebuilt impl of ", n->id(), " does not match its frozen post-op list");
    }
    for (const auto& a : plan.actions) {
        if (a.why == war_downgrade_reason::none) {
            continue;
        }
        const auto& n = p.get_node(a.writer->id());
        for (const auto& e : n.get_fused_primitives_onednn()) {
            const bool still_sum = (e.op_type == onednn_post_op_type::sum || e.op_type == onednn_post_op_type::optimized_sum);
            const bool still_mul = e.op_type == onednn_post_op_type::binary_mul_inplace;
            OPENVINO_ASSERT(!((a.origin == edge_origin::onednn_mul ? still_mul : still_sum) && e.mem_dep == static_cast<size_t>(a.mem_dep)),
                            "[GPU] 1a I-1a-4: ", n.id(), " still holds a frozen in-place post-op at the downgraded dependency ", a.mem_dep);
        }
        if (a.origin == edge_origin::onednn_mul && n.get_selected_impl() != nullptr && n.get_selected_impl()->is_onednn()) {
            // the compiled slot of the vetoed mul is a plain binary_mul now
            const auto slots = get_compiled_post_op_slots(n.get_fused_primitives_onednn(), static_cast<size_t>(a.mem_dep), {onednn_post_op_type::binary_mul});
            OPENVINO_ASSERT(!slots.slot || !n.get_selected_impl()->is_binary_mul_inplace_post_op(slots.kinds, *slots.slot),
                            "[GPU] 1a I-1a-4: the rebuilt impl of ", n.id(), " still compiles a binary_mul_inplace at the vetoed slot");
        }
    }
#    endif
#else
    (void)p;
    (void)plan;
    (void)rep;
#endif  // ENABLE_ONEDNN_FOR_GPU
}

void refreeze_onednn_inplace_post_ops::run(program& p) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::vector<inplace_candidate> candidates;
    for (const auto& node_ptr : p.get_processing_order()) {
        auto& node = *node_ptr;
        if (node.is_dynamic() || node.get_fused_primitives_onednn().empty()) {
            continue;
        }
        std::vector<size_t> frozen_sum_deps;
        for (const auto& e : node.get_fused_primitives_onednn()) {
            if (e.op_type == onednn_post_op_type::sum || e.op_type == onednn_post_op_type::optimized_sum) {
                frozen_sum_deps.push_back(e.mem_dep);
            }
        }
        bool downgraded = false;
        const bool is_onednn_impl = node.get_selected_impl() && node.get_selected_impl()->is_onednn();
        for (const auto mem_dep : frozen_sum_deps) {
            const auto reason = onednn_eltwise_fusing_helpers::frozen_sum_downgrade_reason(node, static_cast<int32_t>(mem_dep));
            if (reason == onednn_eltwise_fusing_helpers::sum_downgrade_reason::none) {
                continue;
            }
            GPU_DEBUG_LOG << "[F5] downgrade in-place sum of " << node.id() << " (dep " << mem_dep << ") to binary_add, reason "
                          << static_cast<int>(reason) << std::endl;
            if (is_onednn_impl) {
                downgraded |= node.downgrade_onednn_sum_to_binary_add(mem_dep);
            } else {
                // B1: non-oneDNN node, the list is a label only; do not build an md / attrs / impl.
                node.relabel_frozen_sum_as_binary_add(mem_dep);
            }
        }
        if (downgraded && is_onednn_impl) {
            rebuild_onednn_impl(p, node, "downgrading the in-place sum to binary_add");
        }
        // Phase 1a: what is still in place after F5 is the candidate set of resolve_inplace_conflicts (same traversal, no extra pass)
        for (const auto& e : node.get_fused_primitives_onednn()) {
            if (e.op_type == onednn_post_op_type::sum || e.op_type == onednn_post_op_type::optimized_sum) {
                candidates.push_back({&node, static_cast<int32_t>(e.mem_dep), edge_origin::onednn_sum});
            } else if (e.op_type == onednn_post_op_type::binary_mul_inplace) {
                candidates.push_back({&node, static_cast<int32_t>(e.mem_dep), edge_origin::onednn_mul});
            }
        }
    }

    // Phase 1a resolve_inplace_conflicts: the single WAR detection point (design 17). Pure plan over war_safe, then one downgrade + rebuild
    // per node. Only downgrades; never inserts a copy.
    auto report = std::make_unique<inplace_conflict_report>();
    report->candidates = candidates.size();
    report->skipped_reason = program_level_skip_reason(p, candidates);
    if (!report->skipped_reason.empty()) {
        if (report->skipped_reason == "internal" || report->skipped_reason == "body" || report->skipped_reason == "oooq") {
            report->program_level_skipped = candidates.size();
        }
    } else {
        inplace_conflict_plan plan;
        try {
            OPENVINO_ASSERT(!take_inplace_conflict_fault_for_test(), "[GPU] 1a: injected analysis failure (test hook)");
            const auto aa = alias_analysis::build(p);   // compile_time_envelope, no hypothetical edges
            const auto axis = exec_axis::build(p);
            if (!axis.valid()) {
                OPENVINO_DEBUG_ASSERT(false, "[GPU] 1a: in_order program without a normalized exec axis after F9");
                plan = downgrade_all(candidates, war_downgrade_reason::no_axis);
            } else {
                plan = plan_inplace_conflict_resolution(aa, axis);
            }
        } catch (...) {
#    ifndef NDEBUG
            throw;
#    else
            GPU_DEBUG_LOG << "[1a] analysis failed, downgrading every in-place candidate" << std::endl;
            plan = downgrade_all(candidates, war_downgrade_reason::analysis_exception);
#    endif
        }
        apply_inplace_conflict_plan(p, plan, *report);
    }
    GPU_DEBUG_LOG << report->summary() << std::endl;
    p._inplace_conflict_report = std::move(report);

#    ifdef F5_DEBUG_CHECKS
    // D1: I-F5-1 holds for every static node: a frozen sum survives only on a oneDNN impl and only if the decision function agrees.
    for (const auto& node_ptr : p.get_processing_order()) {
        const auto& node = *node_ptr;
        if (node.is_dynamic()) {
            continue;
        }
        for (const auto& e : node.get_fused_primitives_onednn()) {
            if (e.op_type != onednn_post_op_type::sum && e.op_type != onednn_post_op_type::optimized_sum) {
                continue;
            }
            const auto reason = onednn_eltwise_fusing_helpers::frozen_sum_downgrade_reason(node, static_cast<int32_t>(e.mem_dep));
            OPENVINO_ASSERT(reason == onednn_eltwise_fusing_helpers::sum_downgrade_reason::none,
                            "[GPU] F5 D1: node ", node.id(), " keeps a frozen sum on dep ", e.mem_dep, " but the decision function says reason ",
                            static_cast<int>(reason));
        }
    }
#    endif  // F5_DEBUG_CHECKS
#else
    (void)p;
#endif  // ENABLE_ONEDNN_FOR_GPU
}
