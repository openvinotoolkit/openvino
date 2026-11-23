// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct MegaKernelImpl : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::megakernel")
    explicit MegaKernelImpl(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        // Only f32 outputs, bfyx format – always valid for this PoC
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
