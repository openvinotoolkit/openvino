// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "openvino/core/except.hpp"

#include <string>
#include <vector>

namespace cldnn {
namespace ocl {

// HW-free kernel builder for the offline compile-only engine. The offline kernels_cache path compiles
// via ocloc and wraps the result in offline_kernel placeholders; it never calls build_kernels. This
// builder exists only so engine::create_kernel_builder() can return a non-null builder without a
// cl::Context. Any call means a non-offline code path is compiling kernels on the compile-only engine.
class offline_kernel_builder : public kernel_builder {
public:
    void build_kernels(const void* /*src*/,
                       size_t /*src_bytes*/,
                       KernelFormat /*src_format*/,
                       const std::string& /*options*/,
                       std::vector<kernel::ptr>& /*out*/) const override {
        OPENVINO_THROW("[GPU offline] kernel_builder::build_kernels is not available on the compile-only "
                       "engine (offline batches are compiled by ocloc into offline_kernel placeholders)");
    }
};

}  // namespace ocl
}  // namespace cldnn
