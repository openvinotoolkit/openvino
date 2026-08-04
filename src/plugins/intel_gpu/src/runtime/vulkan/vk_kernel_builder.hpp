// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"

#include "vk_common.hpp"

namespace cldnn {
namespace vk {

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
//
// clspv must be linked into the plugin as a library (libclspv_core + LLVM),
// see the CLSPV_EXTERNAL_LIBCLC_DIR / CLSPV_LLVM_SOURCE_DIR CMake options.
class vk_kernel_builder : public kernel_builder {
public:
    explicit vk_kernel_builder(VkDevice device) : _device(device) {}
    ~vk_kernel_builder() override = default;

    void build_kernels(const void* src,
                       size_t src_bytes,
                       KernelFormat src_format,
                       const std::string& options,
                       std::vector<kernel::ptr>& out) const override;

private:
    void build_from_spirv(const void* src, size_t src_bytes, std::vector<kernel::ptr>& out) const;

    VkDevice _device;
};

}  // namespace vk
}  // namespace cldnn
