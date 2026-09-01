// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_engine;

class vulkan_kernel_builder final : public kernel_builder {
public:
    explicit vulkan_kernel_builder(const vulkan_engine& engine) : _engine(engine) {}

    void build_kernels(const void* source,
                       size_t source_size,
                       KernelFormat source_format,
                       const std::string& options,
                       std::vector<kernel::ptr>& output) const override;
    void build_kernels(const kernel_artifact& artifact, std::vector<kernel::ptr>& output) const override;

private:
    const vulkan_engine& _engine;
};

}  // namespace vulkan
}  // namespace cldnn
