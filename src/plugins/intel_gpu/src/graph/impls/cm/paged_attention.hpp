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

        // PA CM kernel requires kv_head_size divisible by 16
        if (desc->k_head_size % 16 != 0 && desc->v_head_size % 16 != 0) {
            OPENVINO_ASSERT(!explicit_cm,
                "[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel "
                "requires k_head_size or v_head_size to be divisible by 16. Observed k_head_size=",
                desc->k_head_size, ", v_head_size=", desc->v_head_size,
                ". Set ov::hint::attn_kernel_mode=AUTO to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false because kv_head_size is not divisible by 16. " << std::endl;
            return false;
        }

        auto& engine = node.get_program().get_engine();
        const auto& info = engine.get_device_info();
        // CM optimized for systolic-array architectures
        if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
            OPENVINO_ASSERT(!explicit_cm,
                "[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel "
                "is not supported on this GPU architecture. The CM kernel requires a systolic-array "
                "GPU with immad support and CM JIT enabled. "
                "Set ov::hint::attn_kernel_mode=AUTO to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported GPU architecture. " << std::endl;
            return false;
        }

        const auto& q_layout = node.get_input_layout(PagedAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(PagedAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(PagedAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            OPENVINO_ASSERT(!explicit_cm,
                "[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel "
                "requires bfyx layout for Q/K/V/output. Observed formats: q=", q_layout.format,
                ", k=", k_layout.format, ", v=", v_layout.format, ", out=", out_layout.format,
                ". Set ov::hint::attn_kernel_mode=AUTO to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported qkv layout. " << std::endl;
            return false;
        }

        if (!one_of(k_layout.data_type, supported_q_types) || !one_of(v_layout.data_type, supported_q_types)) {
            OPENVINO_ASSERT(!explicit_cm,
                "[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel "
                "only supports K/V data types {f16}. Observed: k=", k_layout.data_type,
                ", v=", v_layout.data_type,
                ". Set ov::hint::attn_kernel_mode=AUTO to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported kv data type. " << std::endl;
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            OPENVINO_ASSERT(!explicit_cm,
                "[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel "
                "only supports Q/output data types {f16}. Observed: q=", q_layout.data_type,
                ", out=", out_layout.data_type,
                ". Set ov::hint::attn_kernel_mode=AUTO to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported q/out data type. " << std::endl;
            return false;
        }

        const auto& kcache_layout = node.get_input_layout(PagedAttentionInputIdx::KEY_CACHE);
        const auto& vcache_layout = node.get_input_layout(PagedAttentionInputIdx::VALUE_CACHE);
        if (!one_of(kcache_layout.data_type, supported_kv_types) || !one_of(vcache_layout.data_type, supported_kv_types)) {
            OPENVINO_ASSERT(!explicit_cm,
                "[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel "
                "only supports KV cache data types {f16, i8}. Observed: k_cache=", kcache_layout.data_type,
                ", v_cache=", vcache_layout.data_type,
                ". Set ov::hint::attn_kernel_mode=AUTO to allow automatic fallback to a supported kernel.");
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported kv cache data type. " << std::endl;
            return false;
        }

        GPU_DEBUG_TRACE_DETAIL << "validate_impl() - true" << std::endl;
        return true;
    }
};
}  // namespace ov::intel_gpu::cm