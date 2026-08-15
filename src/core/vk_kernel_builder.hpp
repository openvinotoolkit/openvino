// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_common.hpp"
#include "vk_kernel.hpp"
#include "vk_platform.hpp"
#include "vk_types.hpp"

#include <string_view>

namespace ov::core::vulkan {
namespace cross_platform {

// Builds Vulkan compute pipelines from OpenCL C source (KernelFormat::SOURCE)
// or precompiled SPIR-V (KernelFormat::NATIVE_BIN).
//
// SOURCE format: the source is compiled to SPIR-V with clspv
// (https://github.com/google/clspv), which translates OpenCL C into
// Vulkan-compatible SPIR-V (GLCompute execution model, storage buffers,
// push constants). One pipeline per kernel entry point is created.
//
// NATIVE_BIN format: |src| is a raw SPIR-V module; a pipeline is created for
// every entry point found in the module.
class vk_kernel_builder {
public:
    vk_kernel_builder(VkDevice device, const vk_platform_config& config = {});
    ~vk_kernel_builder();

    vk_kernel_builder(const vk_kernel_builder&) = delete;
    vk_kernel_builder& operator=(const vk_kernel_builder&) = delete;

    void build_kernels(const void* src,
                       size_t src_bytes,
                       KernelFormat src_format,
                       const std::string& options,
                       std::vector<vk_kernel_ptr>& out) const;

    // Builds the native kernel with the given id from the builtin SPIR-V
    // store (kernels/*.comp compiled at build time). Throws ov::Exception when
    // the id is unknown.
    void build_native_kernel(const std::string& id, std::vector<vk_kernel_ptr>& out) const;

private:
    void build_from_spirv(const void* src,
                          size_t src_bytes,
                          std::vector<vk_kernel_ptr>& out,
                          std::string_view artifact_base = {}) const;

    VkDevice _device;
    // Vulkan SC offline mode: pipelines are built against this cache and the
    // cache blob is flushed to disk in the destructor.
    VkPipelineCache _pipeline_cache = VK_NULL_HANDLE;
    vk_platform_config _config;
};

}  // namespace cross_platform
}  // namespace ov::core::vulkan
