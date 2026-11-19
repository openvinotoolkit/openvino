// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "program_helpers.h"
#include "primitive_type.h"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <impls/onednn/utils.hpp>
#include "intel_gpu/graph/serialization/weights_reorder_params.hpp"
#endif

using namespace cldnn;

// F5 Debug assertions D1/D2 (zero Release cost). Temporarily define F5_DEBUG_CHECKS to 1 to exercise them in an NDEBUG build.
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

// Re-creates the oneDNN impl of `node` from the patched attributes. Same impl type only (C4); fails loudly when the new impl expects a
// weights layout other than the one the weights dependency was already reordered to (C2: no reorder insertion at this stage, it runs
// after propagate_constants, so a freshly inserted constant reorder would be neither folded nor given an impl).
void rebuild_onednn_impl(program& p, program_node& node) {
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
            OPENVINO_THROW("[GPU] F5: cannot re-create the oneDNN impl of ", node.id(), " after downgrading its sum to binary_add: ", e.what());
        }
    }
    OPENVINO_ASSERT(new_impl && new_impl->is_onednn(),
                    "[GPU] F5: re-creating the impl of ", node.id(), " after downgrading its sum to binary_add did not give a oneDNN impl");

    if (!p.is_internal_program()) {
        const auto new_weights_md = expected_weights_md(*new_impl);
        const bool same = (!old_weights_md && !new_weights_md) || (old_weights_md && new_weights_md && *old_weights_md == *new_weights_md);
        OPENVINO_ASSERT(same,
                        "[GPU] F5: downgrading the in-place sum of ", node.id(),
                        " to binary_add changes the weights layout the oneDNN impl expects; the weights dependency was reordered for the old impl");
    }

    node.set_selected_impl(std::move(new_impl));
    auto* impl = node.get_selected_impl();
    auto new_params = node.get_kernel_impl_params();
    p.get_kernels_cache().add_kernels_source(*new_params, impl->get_kernels_source());
}
}  // namespace
#endif  // ENABLE_ONEDNN_FOR_GPU

void refreeze_onednn_inplace_post_ops::run(program& p) {
#ifdef ENABLE_ONEDNN_FOR_GPU
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
            rebuild_onednn_impl(p, node);
        }
    }
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
