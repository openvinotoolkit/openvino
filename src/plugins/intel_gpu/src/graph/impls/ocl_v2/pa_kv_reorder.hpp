// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <utility>

#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;

namespace ov::intel_gpu::ocl {

struct PA_KV_reorder : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::pa_kv_reorder")
    explicit PA_KV_reorder(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        if (node.has_fused_primitives()) {
            return false;
        }

        const auto desc = node.as<cldnn::pa_kv_reorder>().get_primitive();
        const auto& config = node.get_program().get_config();
        const auto& attn_modes = config.get_attn_mode();
        return !desc->has_xattention && std::find(attn_modes.begin(), attn_modes.end(), ov::hint::AttnMode::PA_CM) == attn_modes.end();
    }
};

}  // namespace ov::intel_gpu::ocl
