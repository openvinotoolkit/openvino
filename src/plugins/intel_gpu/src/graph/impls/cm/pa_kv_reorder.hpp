// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

struct PaKVReorderImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::pa_kv_reorder")
    explicit PaKVReorderImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        if (node.has_fused_primitives()) {
            return false;
        }
        const auto desc = node.as<cldnn::pa_kv_reorder>().get_primitive();
        // CM reorder kernel handles only fp16 / int8 caches (matches CM PA validate_impl).
        if (!(desc->cache_dt == cldnn::data_types::f16 || desc->cache_dt == cldnn::data_types::i8 || desc->cache_dt == cldnn::data_types::u8)) {
            return false;
        }
        // CM PA materializes a token-major KV layout for sparse-attention pipelines
        // (XAttention / qq_bias). The OCL reorder handles the legacy head-major layout.
        // The flag is set in ops/pa_kv_reorder.cpp from the model rt_info "sparse_enabled".
        return desc->is_sparse;
    }
};

}  // namespace ov::intel_gpu::cm
