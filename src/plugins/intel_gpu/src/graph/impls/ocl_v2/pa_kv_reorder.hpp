// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
        if (node.has_fused_primitives())
            return false;
        // OCL kernel handles the legacy head-major KV layout (dense PA). Sparse-attention
        // models (XAttention / qq_bias) use a token-major layout produced by the CM pipeline
        // and are handled by the CM reorder impl. The flag is set in ops/pa_kv_reorder.cpp
        // from the model rt_info "sparse_enabled".
        const auto desc = node.as<cldnn::pa_kv_reorder>().get_primitive();
        return !desc->is_sparse;
    }
};

}  // namespace ov::intel_gpu::ocl
