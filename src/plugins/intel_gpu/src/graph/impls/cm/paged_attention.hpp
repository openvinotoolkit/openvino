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

        // Enable CM PA only in case of XAttention been enabled. May decouple them in future.
        auto desc = node.as<paged_attention>().get_primitive();
        if (!desc->has_xattention) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false because we enable CM PA when XAttention is enabled. " << std::endl;
            return false;
        }

        // PA CM kernel only supports cases when kv_head_size is divisible by 16
        if (desc->k_head_size % 16 != 0 && desc->v_head_size % 16 != 0) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false because kv_head_size is not divisible by 16. " << std::endl;
            return false;
        }

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();
        // CM optimized for systolic-array architectures
        if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported GPU architecture. " << std::endl;
            return false;
        }

        const auto& q_layout = node.get_input_layout(PagedAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(PagedAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(PagedAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported qkv layout. " << std::endl;
            return false;
        }

        if (!one_of(k_layout.data_type, supported_q_types) || !one_of(v_layout.data_type, supported_q_types)) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported kv data type. " << std::endl;
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported q/out data type. " << std::endl;
            return false;
        }

        const auto& kcache_layout = node.get_input_layout(PagedAttentionInputIdx::KEY_CACHE);
        const auto& vcache_layout = node.get_input_layout(PagedAttentionInputIdx::VALUE_CACHE);
        if (!one_of(kcache_layout.data_type, supported_kv_types) || !one_of(vcache_layout.data_type, supported_kv_types)) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false due to unsupported kv cache data type. " << std::endl;
            return false;
        }

        GPU_DEBUG_TRACE_DETAIL << "validate_impl() - true" << std::endl;
        return true;
    }
};
}  // namespace ov::intel_gpu::cm