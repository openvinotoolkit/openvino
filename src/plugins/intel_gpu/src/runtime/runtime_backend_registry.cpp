// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

#include <algorithm>
#include <iterator>

#include "openvino/core/except.hpp"

#if defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_ZE_RT)
#    include "ocl/ocl_device_detector.hpp"
#endif

#ifdef OV_GPU_WITH_OCL_RT
#    include "ocl/ocl_engine_factory.hpp"
#endif

#ifdef OV_GPU_WITH_ZE_RT
#    include "ze/ze_device_detector.hpp"
#    include "ze/ze_engine_factory.hpp"
#endif

#ifdef OV_GPU_WITH_SYCL_RT
#    include "sycl/sycl_device_detector.hpp"
#    include "sycl/sycl_engine_factory.hpp"
#endif

#ifdef OV_GPU_WITH_VULKAN_RT
#    include "vulkan/vulkan_device_detector.hpp"
#    include "vulkan/vulkan_engine_factory.hpp"
#endif

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
        break;
    }
    OPENVINO_THROW("[GPU] Unknown runtime type");
}

runtime_types compiled_default_runtime_type() {
#ifdef OV_GPU_DEFAULT_ZE_RT
    return runtime_types::ze;
#elif defined(OV_GPU_DEFAULT_OCL_RT)
    return runtime_types::ocl;
#elif defined(OV_GPU_DEFAULT_SYCL_RT)
    return runtime_types::sycl;
#elif defined(OV_GPU_DEFAULT_VULKAN_RT)
    return runtime_types::vulkan;
#else
#    error "Expected an OpenVINO default GPU runtime macro to be defined"
#endif
}

}  // namespace

const std::vector<runtime_backend_descriptor>& runtime_backend_registry::compiled_backends() {
    static const std::vector<runtime_backend_descriptor> backends = [] {
        std::vector<runtime_backend_descriptor> result;
#ifdef OV_GPU_WITH_OCL_RT
        result.push_back({engine_types::ocl, runtime_types::ocl, "ocl", {}, {}});
#endif
#ifdef OV_GPU_WITH_ZE_RT
        result.push_back({engine_types::ze, runtime_types::ze, "ze", {}, {}});
#endif
#ifdef OV_GPU_WITH_SYCL_RT
        result.push_back({engine_types::sycl, runtime_types::sycl, "sycl", {}, {}});
#endif
#ifdef OV_GPU_WITH_VULKAN_RT
        gpu_operation_lowering_capabilities vulkan_operation_lowering;
        vulkan_operation_lowering.direct_divide = true;
        vulkan_operation_lowering.direct_binary_power = true;
        gpu_kernel_cache_policy vulkan_kernel_cache;
        vulkan_kernel_cache.artifact = gpu_cached_kernel_artifact::spirv;
        result.push_back({engine_types::vulkan, runtime_types::vulkan, "vulkan", vulkan_operation_lowering, vulkan_kernel_cache});
#endif
        const auto default_runtime = compiled_default_runtime_type();
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

std::map<std::string, std::shared_ptr<device>> runtime_backend_registry::query_devices(engine_types engine_type,
                                                                                       runtime_types runtime_type,
                                                                                       void* user_context,
                                                                                       void* user_device,
                                                                                       int context_device_id,
                                                                                       int target_tile_id,
                                                                                       bool initialize_devices) {
    switch (runtime_type) {
#if defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_ZE_RT)
    case runtime_types::ocl: {
        OPENVINO_ASSERT(engine_type == engine_types::ocl || engine_type == engine_types::sycl);
        ocl::ocl_device_detector detector;
        auto devices = detector.get_available_devices(user_context, user_device, context_device_id, target_tile_id, initialize_devices);
#    ifdef OV_GPU_WITH_ZE_RT
        if (compiled_default_runtime_type() == runtime_types::ze) {
            for (auto& device : devices) {
                device.second = ze::create_ze_device_from_ocl_device(device.second, initialize_devices);
            }
        }
#    endif
        return devices;
    }
#endif
#ifdef OV_GPU_WITH_ZE_RT
    case runtime_types::ze: {
        OPENVINO_ASSERT(engine_type == engine_types::ze);
        ze::ze_device_detector detector;
        return detector.get_available_devices(user_context, user_device, context_device_id, target_tile_id, initialize_devices);
    }
#endif
#ifdef OV_GPU_WITH_SYCL_RT
    case runtime_types::sycl: {
        OPENVINO_ASSERT(engine_type == engine_types::sycl);
        sycl::sycl_device_detector detector;
        return detector.get_available_devices(user_context, user_device, context_device_id, target_tile_id);
    }
#endif
#ifdef OV_GPU_WITH_VULKAN_RT
    case runtime_types::vulkan: {
        OPENVINO_ASSERT(engine_type == engine_types::vulkan);
        vulkan::vulkan_device_detector detector;
        return detector.get_available_devices(user_context, user_device, context_device_id, target_tile_id, initialize_devices);
    }
#endif
    default:
        break;
    }
    OPENVINO_THROW("[GPU] Unsupported engine/runtime types in device query");
}

std::shared_ptr<engine> runtime_backend_registry::create_engine(engine_types engine_type, runtime_types runtime_type, const std::shared_ptr<device>& device) {
    switch (engine_type) {
#ifdef OV_GPU_WITH_SYCL_RT
    case engine_types::sycl:
        return sycl::create_sycl_engine(device, runtime_type);
#endif
#ifdef OV_GPU_WITH_OCL_RT
#    if defined(OV_GPU_WITH_SYCL) && !defined(OV_GPU_WITH_SYCL_RT)
    case engine_types::sycl:
        return ocl::create_sycl_engine(device, runtime_type);
#    endif
    case engine_types::ocl:
        return ocl::create_ocl_engine(device, runtime_type);
#endif
#ifdef OV_GPU_WITH_ZE_RT
    case engine_types::ze:
        return ze::create_ze_engine(device, runtime_type);
#endif
#ifdef OV_GPU_WITH_VULKAN_RT
    case engine_types::vulkan:
        return vulkan::create_vulkan_engine(device, runtime_type);
#endif
    default:
        break;
    }
    OPENVINO_THROW("[GPU] Unsupported engine type");
}

std::string runtime_backend_registry::make_device_id(runtime_types runtime_type, const std::string& backend_device_id) {
    return std::string(runtime_name(runtime_type)) + "_" + backend_device_id;
}

std::string runtime_backend_registry::make_public_device_id(runtime_types runtime_type, const std::string& backend_device_id) {
    if (runtime_type == default_backend().runtime_type) {
        return backend_device_id;
    }
    return make_device_id(runtime_type, backend_device_id);
}

std::string runtime_backend_registry::select_default_device_id(const std::vector<std::pair<runtime_types, std::string>>& available_devices) {
    const auto default_runtime = default_backend().runtime_type;
    const auto device_it = std::find_if(available_devices.begin(), available_devices.end(), [default_runtime](const auto& device) {
        return device.first == default_runtime;
    });
    if (device_it == available_devices.end()) {
        return {};
    }
    return make_public_device_id(device_it->first, device_it->second);
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
