// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace cldnn {

enum class gpu_arithmetic_support : uint8_t {
    unavailable,
    emulated,
    native,
};

struct gpu_numeric_capability {
    bool storage = false;
    gpu_arithmetic_support arithmetic = gpu_arithmetic_support::unavailable;
};

enum class gpu_layout_kind : uint8_t {
    dense_buffer,
    strided_buffer,
    blocked_buffer,
    image,
    planar_image,
};

struct gpu_layout_capabilities {
    bool dense_buffers = false;
    bool strided_buffers = false;
    bool blocked_buffers = false;
    bool images = false;
    bool planar_images = false;
    uint32_t max_tensor_rank = 0;

    bool supports(gpu_layout_kind kind) const noexcept {
        switch (kind) {
        case gpu_layout_kind::dense_buffer:
            return dense_buffers;
        case gpu_layout_kind::strided_buffer:
            return strided_buffers;
        case gpu_layout_kind::blocked_buffer:
            return blocked_buffers;
        case gpu_layout_kind::image:
            return images;
        case gpu_layout_kind::planar_image:
            return planar_images;
        }
        return false;
    }
};

struct gpu_synchronization_capabilities {
    bool synchronization2 = false;
    bool timeline_semaphores = false;
};

struct gpu_external_memory_capabilities {
    bool host_pointer = false;
    bool dma_buf = false;
    bool android_hardware_buffer = false;
    bool metal_buffer = false;
};

struct gpu_operation_capabilities {
    bool direct_divide = false;
    bool direct_binary_power = false;
};

enum class gpu_cached_kernel_artifact : uint8_t {
    native_device_binary,
    spirv,
};

struct gpu_kernel_cache_capabilities {
    static constexpr uint32_t current_artifact_schema_version = 1;

    gpu_cached_kernel_artifact artifact = gpu_cached_kernel_artifact::native_device_binary;
    uint32_t artifact_schema_version = current_artifact_schema_version;
};

/// Backend-neutral capabilities selected once during device initialization.
struct gpu_backend_capabilities {
    gpu_numeric_capability fp16;
    gpu_numeric_capability fp32;
    gpu_numeric_capability fp64;
    gpu_numeric_capability int8;
    bool subgroup_operations = false;
    uint32_t subgroup_size = 0;
    bool specialization_constants = false;
    bool local_memory = false;
    gpu_synchronization_capabilities synchronization;
    gpu_external_memory_capabilities external_memory;
    gpu_operation_capabilities operations;
    gpu_kernel_cache_capabilities kernel_cache;
    gpu_layout_capabilities layouts;
    bool persistent_pipeline_cache = false;
};

}  // namespace cldnn
