// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace cldnn {

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
    gpu_kernel_cache_capabilities kernel_cache;
};

}  // namespace cldnn
