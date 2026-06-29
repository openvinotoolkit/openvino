// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

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
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
