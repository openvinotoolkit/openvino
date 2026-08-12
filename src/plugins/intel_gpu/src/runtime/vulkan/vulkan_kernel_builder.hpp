// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_kernel_builder final : public kernel_builder {
public:
    void build_kernels(const void* source,
                       size_t source_size,
                       KernelFormat source_format,
                       const std::string& options,
                       std::vector<kernel::ptr>& output) const override;
};

}  // namespace vulkan
}  // namespace cldnn
