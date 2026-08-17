// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

#include <algorithm>
#include <iterator>

#include "openvino/core/except.hpp"

namespace cldnn {
namespace {

const char* runtime_name(runtime_types runtime_type) {
    switch (runtime_type) {
    case runtime_types::ocl:
        return "ocl";
    case runtime_types::ze:
        return "ze";
    case runtime_types::sycl:
        return "sycl";
    case runtime_types::vulkan:
        return "vulkan";
    default:
        OPENVINO_THROW("[GPU] Unknown runtime type");
    }
}

}  // namespace

const std::vector<runtime_backend_descriptor>& runtime_backend_registry::compiled_backends() {
    static const std::vector<runtime_backend_descriptor> backends = [] {
        std::vector<runtime_backend_descriptor> result;
#ifdef OV_GPU_WITH_OCL_RT
        result.push_back({engine_types::ocl, runtime_types::ocl, "ocl"});
#endif
#ifdef OV_GPU_WITH_ZE_RT
        result.push_back({engine_types::ze, runtime_types::ze, "ze"});
#endif
#ifdef OV_GPU_WITH_SYCL_RT
        result.push_back({engine_types::sycl, runtime_types::sycl, "sycl"});
#endif
#ifdef OV_GPU_WITH_VULKAN_RT
        result.push_back({engine_types::vulkan, runtime_types::vulkan, "vulkan"});
#endif
        const auto default_runtime = get_default_runtime_type();
        const auto default_it = std::find_if(result.begin(), result.end(), [default_runtime](const auto& backend) {
            return backend.runtime_type == default_runtime;
        });
        OPENVINO_ASSERT(default_it != result.end(), "[GPU] Default runtime is not compiled into the plugin");
        std::rotate(result.begin(), default_it, std::next(default_it));
        return result;
    }();
    return backends;
}

const runtime_backend_descriptor& runtime_backend_registry::default_backend() {
    const auto& backends = compiled_backends();
    OPENVINO_ASSERT(!backends.empty(), "[GPU] No runtime backend is compiled into the plugin");
    return backends.front();
}

const runtime_backend_descriptor& runtime_backend_registry::get(runtime_types runtime_type) {
    const auto& backends = compiled_backends();
    const auto it = std::find_if(backends.begin(), backends.end(), [runtime_type](const auto& backend) {
        return backend.runtime_type == runtime_type;
    });
    OPENVINO_ASSERT(it != backends.end(), "[GPU] Requested runtime is not compiled into the plugin: ", runtime_name(runtime_type));
    return *it;
}

std::string runtime_backend_registry::make_device_id(runtime_types runtime_type, const std::string& backend_device_id) {
    return std::string(runtime_name(runtime_type)) + "_" + backend_device_id;
}

bool runtime_backend_registry::parse_device_id(const std::string& device_id, runtime_types& runtime_type, std::string& backend_device_id) {
    const auto separator = device_id.find('_');
    if (separator == std::string::npos || separator == 0 || separator + 1 == device_id.size()) {
        return false;
    }

    const auto prefix = device_id.substr(0, separator);
    for (const auto& backend : compiled_backends()) {
        if (prefix == backend.name) {
            runtime_type = backend.runtime_type;
            backend_device_id = device_id.substr(separator + 1);
            return true;
        }
    }
    return false;
}

}  // namespace cldnn
