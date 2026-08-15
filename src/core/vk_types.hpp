// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_types: the core-owned runtime types of the self-contained Vulkan layer.
//
// The Vulkan core (ov::core::vulkan::cross_platform) no longer reuses the
// intel_gpu / cldnn runtime interfaces. Everything the backend needs —
// allocation kinds, device properties, tensor layouts and kernel argument
// descriptors — is defined here, in the core's own namespace.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/uuid.hpp"

namespace ov::core::vulkan {
namespace cross_platform {

// How a vk_memory allocation is meant to be accessed.
enum class allocation_type {
    unknown,
    usm_host,     // host-visible + host-coherent, permanently mapped
    usm_shared,   // host-visible, device-accessible
    usm_device,   // device-local
};

enum class device_type {
    integrated_gpu,
    discrete_gpu,
};

enum class gpu_arch {
    unknown,
};

enum class engine_types {
    vulkan,
};

enum class runtime_types {
    vulkan,
};

enum class backend_types {
    vulkan,
};

// Kernel source format accepted by vk_kernel_builder.
enum class KernelFormat {
    SOURCE,      // OpenCL C, compiled with clspv (OV_GPU_WITH_CLSPV)
    NATIVE_BIN,  // precompiled SPIR-V module
};

// Tensor layout: logical element dimensions + element size in bytes.
// The Vulkan core only allocates buffers, so a format/shape specialization
// beyond this is not needed (f32 buffers; NCHW memory is implicit).
struct layout {
    std::vector<size_t> shape;
    size_t element_size = 4;

    layout() = default;
    layout(std::vector<size_t> s, size_t element_size = 4)
        : shape(std::move(s)), element_size(element_size) {}

    size_t element_count() const {
        size_t n = 1;
        for (const size_t d : shape)
            n *= d;
        return n;
    }
    size_t byte_size() const { return element_count() * element_size; }
};

// Pod scalar values pushed to the shader (mirrors the clspv reflection order).
enum class scalar_t {
    UINT8,
    INT8,
    UINT16,
    INT16,
    UINT32,
    INT32,
    FLOAT32,
    UINT64,
    INT64,
    FLOAT64,
};

struct scalar_desc {
    scalar_t t = scalar_t::FLOAT32;
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        uint64_t u64;
        int64_t i64;
        double f64;
    } v{};
};

using scalars_desc = std::vector<scalar_desc>;

struct work_group_sizes {
    std::array<size_t, 3> local = {1, 1, 1};
    std::array<size_t, 3> global = {1, 1, 1};
};

struct kernel_arguments_desc {
    work_group_sizes workGroups;
    scalars_desc scalars;
};

class vk_memory;

// Buffers bound to a kernel dispatch, in shader binding order. Values are
// const: dispatching never modifies the argument table, only the buffers.
struct kernel_arguments_data {
    std::vector<std::shared_ptr<const vk_memory>> inputs;
    std::vector<std::shared_ptr<const vk_memory>> intermediates;
    std::vector<std::shared_ptr<const vk_memory>> outputs;
    std::shared_ptr<const vk_memory> weights;
    std::shared_ptr<const vk_memory> recurrent;
    std::shared_ptr<const vk_memory> hidden;
    std::shared_ptr<const vk_memory> cell;
    std::shared_ptr<const vk_memory> bias;
    std::shared_ptr<const vk_memory> weights_zero_points;
    std::shared_ptr<const vk_memory> activations_zero_points;
    std::shared_ptr<const vk_memory> compensation;
    std::shared_ptr<const vk_memory> lookup_table;
    std::shared_ptr<const vk_memory> scale_table;
    std::shared_ptr<const vk_memory> slope;
    std::shared_ptr<const vk_memory> shape_info;
    std::vector<std::shared_ptr<const vk_memory>> fused_op_inputs;
    const scalars_desc* scalars = nullptr;
};

// Physical device properties queried at vk_device construction.
struct device_info {
    uint32_t vendor_id = 0;
    std::string dev_name;
    std::string driver_version;
    device_type dev_type = device_type::integrated_gpu;

    size_t max_work_group_size = 0;
    size_t max_local_mem_size = 0;
    size_t max_global_cache_size = 0;
    size_t max_image2d_width = 0;
    size_t max_image2d_height = 0;
    uint64_t max_global_mem_size = 0;
    uint64_t max_alloc_mem_size = 0;

    bool supports_fp16 = false;
    bool supports_fp64 = false;
    bool supports_fp16_denorms = false;
    bool supports_khr_subgroups = false;
    bool supports_intel_subgroups = false;
    bool supports_intel_subgroups_short = false;
    bool supports_intel_subgroups_char = false;
    bool supports_intel_required_subgroup_size = false;
    bool supports_queue_families = false;
    bool supports_image = false;
    bool supports_intel_planar_yuv = false;
    bool supports_work_group_collective_functions = false;
    bool supports_non_uniform_work_group = false;
    bool supports_imad = false;
    bool supports_immad = false;
    bool supports_mutable_command_list = false;
    bool supports_usm = false;
    bool has_separate_cache = false;
    bool supports_cp_offload = false;
    bool supports_counter_based_events = false;
    bool supports_leo = false;

    size_t execution_units_count = 0;
    size_t gpu_frequency = 0;
    std::vector<size_t> supported_simd_sizes;

    ov::device::UUID uuid;
    ov::device::LUID luid;

    std::array<uint32_t, 3> gfx_ver = {0, 0, 0};
    gpu_arch arch = gpu_arch::unknown;
    size_t ip_version = 0;
    uint32_t device_id = 0;
    size_t num_slices = 0;
    size_t num_sub_slices_per_slice = 0;
    size_t num_eus_per_sub_slice = 0;
    size_t num_threads_per_eu = 0;
    size_t num_ccs = 0;
    size_t sub_device_idx = 0;

    size_t timer_resolution = 0;
    size_t kernel_timestamp_valid_bits = 0;
    size_t compute_queue_group_ordinal = 0;
    size_t device_memory_ordinal = 0;
};

}  // namespace cross_platform
}  // namespace ov::core::vulkan
