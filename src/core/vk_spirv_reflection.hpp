// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// SPIR-V reflection for clspv-generated modules.
//
// Parses the SPIR-V produced by clspv, extracting:
//   - kernel entry points (names),
//   - required workgroup size (OpExecutionMode LocalSize / ClspvReflection
//     PropertyRequiredWorkgroupSize),
//   - kernel argument layout from the NonSemantic.ClspvReflection.5 set:
//     storage buffer bindings, pod push constant offsets/sizes, workgroup
//     (__local) arguments, etc.
//
// The NonSemantic.ClspvReflection grammar revision 7 opcodes are used.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

enum class vk_arg_kind {
    storage_buffer,        // ArgumentStorageBuffer (3)
    uniform_buffer,        // ArgumentUniform (4)
    pod_storage_buffer,    // ArgumentPodStorageBuffer (5)
    pod_uniform,           // ArgumentPodUniform (6)
    pod_push_constant,     // ArgumentPodPushConstant (7)
    sampled_image,         // ArgumentSampledImage (8)
    storage_image,         // ArgumentStorageImage (9)
    sampler,               // ArgumentSampler (10)
    workgroup,             // ArgumentWorkgroup (11)
    pointer_push_constant, // ArgumentPointerPushConstant (26)
    pointer_uniform,       // ArgumentPointerUniform (27)
    storage_texel_buffer,  // ArgumentStorageTexelBuffer (34)
    uniform_texel_buffer,  // ArgumentUniformTexelBuffer (35)
};

const char* arg_kind_name(vk_arg_kind kind);

struct vk_kernel_arg {
    uint32_t ordinal = 0;  // index in the kernel signature
    vk_arg_kind kind = vk_arg_kind::storage_buffer;
    uint32_t descriptor_set = 0;
    uint32_t binding = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
    std::string name;
};

struct vk_kernel_reflection {
    std::string name;
    uint32_t local_size[3] = {1, 1, 1};  // required workgroup size (1 if unspecified)
    bool has_local_size = false;
    // True when the module declares the workgroup size via specialization
    // constants (SpecConstantWorkgroupSize, SpecId 0/1/2) instead of literal
    // OpExecutionMode LocalSize. Then EVERY kernel in the module must be
    // specialized at pipeline creation with its local_size[] values, otherwise
    // the driver dispatches with a workgroup size of 1.
    bool uses_spec_wgsize = false;
    std::vector<vk_kernel_arg> args;  // sorted by ordinal

    // Highest storage/uniform buffer binding + 1.
    uint32_t max_binding() const;
    // Total push constant bytes needed (max offset+size over pod push constants).
    uint32_t push_constants_size() const;
};

// Parses |spirv| and returns reflection for all kernels. Returns an empty
// vector if the module cannot be parsed.
std::vector<vk_kernel_reflection> parse_spirv_reflection(const std::vector<uint32_t>& spirv);

}  // namespace cross_platform
}  // namespace ov::core::vulkan
