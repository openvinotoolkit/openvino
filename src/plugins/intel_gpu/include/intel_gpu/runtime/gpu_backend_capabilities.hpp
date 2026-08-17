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
    image,
    planar_image,
};

struct gpu_layout_capabilities {
    bool dense_buffers = false;
    bool strided_buffers = false;
    bool images = false;
    bool planar_images = false;
    uint32_t max_tensor_rank = 0;

    bool supports(gpu_layout_kind kind) const noexcept {
        switch (kind) {
        case gpu_layout_kind::dense_buffer:
            return dense_buffers;
        case gpu_layout_kind::strided_buffer:
            return strided_buffers;
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

enum class gpu_execution_tier : uint8_t {
    legacy,
    portable,
    optimized,
};

/// Backend-neutral capabilities selected once during device initialization.
struct gpu_backend_capabilities {
    bool legacy_device_info_adapter = true;
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
    gpu_layout_capabilities layouts;
    bool persistent_pipeline_cache = false;
    gpu_execution_tier execution_tier = gpu_execution_tier::legacy;
};

}  // namespace cldnn
