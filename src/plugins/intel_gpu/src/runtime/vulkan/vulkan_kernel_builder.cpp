// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel_builder.hpp"

#include <cstdint>
#include <cstring>

#include "openvino/core/except.hpp"
#include "vulkan_clspv_compiler.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_kernel.hpp"
#include "vulkan_kernel_interface.hpp"

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
    artifact.entry_point = source_format == KernelFormat::SOURCE ? "main" : std::string{};
    artifact.build_options = options;
    build_kernels(artifact, output);
}

void vulkan_kernel_builder::build_kernels(const kernel_artifact& artifact, std::vector<kernel::ptr>& output) const {
    OPENVINO_ASSERT(artifact.format == KernelFormat::SOURCE || artifact.format == KernelFormat::SPIRV,
                    "[GPU][Vulkan] Vulkan kernels must be source or SPIR-V artifacts");
    OPENVINO_ASSERT(artifact.payload != nullptr && artifact.payload_size > 0, "[GPU][Vulkan] Cannot build an empty kernel");

    std::vector<uint8_t> binary;
    std::string build_log;
    if (artifact.format == KernelFormat::SOURCE) {
        const std::string source(static_cast<const char*>(artifact.payload), artifact.payload_size);
        auto compilation = vulkan_clspv_compiler{}.compile(source, artifact.build_options, artifact.entry_point);
        binary = std::move(compilation.spirv);
        build_log = std::move(compilation.diagnostics);
    } else {
        binary.resize(artifact.payload_size);
        std::memcpy(binary.data(), artifact.payload, artifact.payload_size);
    }

    const auto entry_point = artifact.entry_point.empty() ? vulkan_kernel_interface::get_single_entry_point(binary) : artifact.entry_point;
    output.push_back(std::make_shared<vulkan_kernel>(_engine.get_vulkan_device_object(), std::move(binary), entry_point, std::move(build_log)));
}

kernel_compiler_info vulkan_kernel_builder::get_compiler_info() const {
    kernel_compiler_info info;
    info.source_cache_format = KernelFormat::SPIRV;
    info.source_headers = KernelSourceHeaders::REFERENCED_ONLY;
    info.max_source_kernels_per_batch = 1;
    info.cache_identity = vulkan_clspv_compiler::identity() + " " + vulkan_clspv_compiler::canonical_options({});
    return info;
}

}  // namespace vulkan
}  // namespace cldnn
