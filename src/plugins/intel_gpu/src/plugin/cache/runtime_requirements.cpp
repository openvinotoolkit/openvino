// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime_requirements.hpp"

#include <sstream>

#include "intel_gpu/runtime/device.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"

namespace ov::intel_gpu::cache {

std::string build_runtime_requirements(const cldnn::device& device) {
    const auto& info = device.get_info();
    const auto artifact = device.get_backend_capabilities().kernel_cache.artifact;
    const char* artifact_name = nullptr;
    switch (artifact) {
    case cldnn::gpu_cached_kernel_artifact::native_device_binary:
        artifact_name = "native_device_binary";
        break;
    case cldnn::gpu_cached_kernel_artifact::spirv:
        artifact_name = "spirv";
        break;
    }
    OPENVINO_ASSERT(artifact_name != nullptr, "[GPU] Unsupported cached kernel artifact format");

    std::ostringstream descriptor;
    descriptor << "meta=" << runtime_requirements_version << ".0";
    descriptor << ";ov=" << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
    descriptor << ";runtime=" << device.get_runtime_type();
    descriptor << ";kernel_artifact=" << artifact_name;
    descriptor << ";artifact_schema=" << device.get_backend_capabilities().kernel_cache.artifact_schema_version;
    descriptor << ";desc=[driver=" << info.driver_version;
    descriptor << ";ip=" << info.gfx_ver.major << "." << static_cast<uint32_t>(info.gfx_ver.minor) << ".";
    descriptor << static_cast<uint32_t>(info.gfx_ver.revision);
    descriptor << ";eus=" << info.execution_units_count << "]";
    return descriptor.str();
}

bool is_runtime_requirements_compatible(const std::string& requirements, const cldnn::device& device) {
    return requirements == build_runtime_requirements(device);
}

}  // namespace ov::intel_gpu::cache
