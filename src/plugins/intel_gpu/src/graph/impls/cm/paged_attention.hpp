// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/runtime/layout.hpp"
#include "paged_attention_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct PagedAttentionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::paged_attention::opt")
    explicit PagedAttentionImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_q_types = {
            ov::element::f16,
        };
        static constexpr std::array supported_kv_types = {
            ov::element::f16,
            ov::element::i8,
        };

        const auto& config = node.get_program().get_config();

        // Enable CM PA for XAttention or when explicitly requested by attention kernel mode hint.
        auto desc = node.as<paged_attention>().get_primitive();
        const bool explicit_cm = desc->use_cm_kernel;

        // CM PA is enabled when XAttention is active, or when explicitly requested via use_cm_kernel.
        if (!desc->has_xattention && !explicit_cm) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false: CM PA requires XAttention or explicit use_cm_kernel. " << std::endl;
            return false;
        }

        if (desc->k_head_size != desc->v_head_size) {
            OPENVINO_ASSERT(!explicit_cm, "[GPU] CM PagedAttention requires equal K/V head sizes.");
            return false;
        }

        if (!desc->has_xattention &&
            (desc->sliding_window != 0 || desc->has_alibi || desc->has_sink_input || desc->has_rotated_blocks ||
             desc->has_token_type_ids || desc->has_scores_output() || desc->has_score_aggregation || desc->has_adaptive_rkv)) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] CM PagedAttention does not support sliding-window attention, ALiBi, attention sinks, "
                            "cache rotation, token-type inputs, scores output, score aggregation, or adaptive RKV. "
                            "Set ov::intel_gpu::kernel_selection to an empty list to use OCL PagedAttention.");
            return false;
        }

        if (!desc->scale_val.has_value()) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] CM PagedAttention requires a constant attention scale. "
                            "Set ov::intel_gpu::kernel_selection to an empty list to use a runtime scale with OCL PagedAttention.");
            return false;
        }

        // PA CM kernel requires kv_head_size divisible by 16
        if (desc->k_head_size % 16 != 0 || desc->v_head_size % 16 != 0) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] ov::intel_gpu::kernel_selection=PA_KERNEL_AUTO selected CM, but the CM paged attention kernel "
                            "requires both k_head_size and v_head_size to be divisible by 16. Observed k_head_size=",
                            desc->k_head_size,
                            ", v_head_size=",
                            desc->v_head_size,
                            ". Set ov::intel_gpu::kernel_selection to an empty list to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false because kv_head_size is not divisible by 16. " << std::endl;
            return false;
        }

        if (!config.is_cm_available()) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] ov::intel_gpu::kernel_selection=PA_KERNEL_AUTO selected CM, but the CM paged attention kernel "
                            "is not supported on this GPU architecture. The CM kernel requires a systolic-array "
                            "GPU with immad support and CM JIT enabled. "
                            "Set ov::intel_gpu::kernel_selection to an empty list to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported GPU architecture. " << std::endl;
            return false;
        }

        const auto& q_layout = node.get_input_layout(PagedAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(PagedAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(PagedAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] ov::intel_gpu::kernel_selection=PA_KERNEL_AUTO selected CM, but the CM paged attention kernel "
                            "requires bfyx layout for Q/K/V/output. Observed formats: q=",
                            q_layout.format,
                            ", k=",
                            k_layout.format,
                            ", v=",
                            v_layout.format,
                            ", out=",
                            out_layout.format,
                            ". Set ov::intel_gpu::kernel_selection to an empty list to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported qkv layout. " << std::endl;
            return false;
        }

        if (!one_of(k_layout.data_type, supported_q_types) || !one_of(v_layout.data_type, supported_q_types)) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] ov::intel_gpu::kernel_selection=PA_KERNEL_AUTO selected CM, but the CM paged attention kernel "
                            "only supports K/V data types {f16}. Observed: k=",
                            k_layout.data_type,
                            ", v=",
                            v_layout.data_type,
                            ". Set ov::intel_gpu::kernel_selection to an empty list to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported kv data type. " << std::endl;
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] ov::intel_gpu::kernel_selection=PA_KERNEL_AUTO selected CM, but the CM paged attention kernel "
                            "only supports Q/output data types {f16}. Observed: q=",
                            q_layout.data_type,
                            ", out=",
                            out_layout.data_type,
                            ". Set ov::intel_gpu::kernel_selection to an empty list to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported q/out data type. " << std::endl;
            return false;
        }

        const auto& kcache_layout = node.get_input_layout(PagedAttentionInputIdx::KEY_CACHE);
        const auto& vcache_layout = node.get_input_layout(PagedAttentionInputIdx::VALUE_CACHE);
        if (!one_of(kcache_layout.data_type, supported_kv_types) || !one_of(vcache_layout.data_type, supported_kv_types)) {
            OPENVINO_ASSERT(!explicit_cm,
                            "[GPU] ov::intel_gpu::kernel_selection=PA_KERNEL_AUTO selected CM, but the CM paged attention kernel "
                            "only supports KV cache data types {f16, i8}. Observed: k_cache=",
                            kcache_layout.data_type,
                            ", v_cache=",
                            vcache_layout.data_type,
                            ". Set ov::intel_gpu::kernel_selection to an empty list to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported kv cache data type. " << std::endl;
            return false;
        }

        GPU_DEBUG_TRACE_DETAIL << "validate_impl() - true" << std::endl;
        return true;
    }
};
}  // namespace ov::intel_gpu::cm