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

// Not used: the offline path compiles with ocloc, so any call here is a bug.
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
