// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel_builder.hpp"

#include <cstdint>
#include <cstring>

#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_kernel.hpp"

namespace cldnn {
namespace vulkan {

void vulkan_kernel_builder::build_kernels(const void* source,
                                          size_t source_size,
                                          KernelFormat source_format,
                                          const std::string& options,
                                          std::vector<kernel::ptr>& output) const {
    kernel_artifact artifact;
    artifact.payload = source;
    artifact.payload_size = source_size;
    artifact.format = source_format;
    artifact.entry_point = "main";
    artifact.build_options = options;
    build_kernels(artifact, output);
}

void vulkan_kernel_builder::build_kernels(const kernel_artifact& artifact, std::vector<kernel::ptr>& output) const {
    OPENVINO_ASSERT(artifact.format == KernelFormat::SPIRV, "[GPU][Vulkan] Vulkan kernels must be tagged as SPIR-V artifacts");
    OPENVINO_ASSERT(artifact.payload != nullptr && artifact.payload_size > 0, "[GPU][Vulkan] Cannot build an empty SPIR-V kernel");

    std::vector<uint8_t> binary(artifact.payload_size);
    std::memcpy(binary.data(), artifact.payload, artifact.payload_size);
    output.push_back(
        std::make_shared<vulkan_kernel>(_engine.get_vulkan_device_object(), std::move(binary), artifact.entry_point.empty() ? "main" : artifact.entry_point));
}

}  // namespace vulkan
}  // namespace cldnn
