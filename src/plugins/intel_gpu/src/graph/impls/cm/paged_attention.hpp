// Copyright (C) 2025 Intel Corporation
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

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();
        // CM optimized for systolic-array architectures
        if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
            GPU_DEBUG_TRACE_DETAIL << __LINE__ << ": ov::intel_gpu::cm::PagedAttentionImplementationManager::validate_impl() - false " << std::endl;
            return false;
        }

        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            GPU_DEBUG_TRACE_DETAIL << __LINE__ << ": ov::intel_gpu::cm::PagedAttentionImplementationManager::validate_impl() - false " << std::endl;
            return false;
        }

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types)) {
            GPU_DEBUG_TRACE_DETAIL << __LINE__ << ": ov::intel_gpu::cm::PagedAttentionImplementationManager::validate_impl() - false " << std::endl;
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            GPU_DEBUG_TRACE_DETAIL << __LINE__ << ": ov::intel_gpu::cm::PagedAttentionImplementationManager::validate_impl() - false " << std::endl;
            return false;
        }

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionImplementationManager::validate_impl() - true" << std::endl;
        return true;
    }
};
}  // namespace ov::intel_gpu::cm