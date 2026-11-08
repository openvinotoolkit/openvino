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
        if (node.has_fused_primitives()) {
            return false;
        }

        const auto desc = node.as<cldnn::pa_kv_reorder>().get_primitive();
        // A 4-bit K cache is BY_CHANNEL only (the token-major page has no 4-bit BY_TOKEN K kind, paged_attention.hpp).
        if (desc->is_kv_compressed && data_type_traits::is_i4_u4(desc->cache_dt) && !desc->is_key_by_channel) {
            return false;
        }
        return !desc->has_xattention;
    }
};

}  // namespace ov::intel_gpu::ocl
