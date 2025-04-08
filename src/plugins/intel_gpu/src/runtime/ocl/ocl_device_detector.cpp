// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_device_detector.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "ocl_device.hpp"
#include "ocl_common.hpp"

#include <string>
#include <vector>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace {
static const char create_device_error_msg[] =
    "[GPU] No supported OCL devices found or unexpected error happened during devices query.\n"
    "[GPU] Please check OpenVINO documentation for GPU drivers setup guide.\n";

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

bool does_device_match_config(const cl::Device& device) {
    if (device.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_GPU) {
        return false;
    }

    int32_t ocl_major = -1;
    int32_t ocl_minor = -1;
    // Spec says that the format of this string is OpenCL<space><major_version.minor_version><space><vendor-specific information>
    auto ocl_version_string = device.getInfo<CL_DEVICE_VERSION>();
    auto tokens = split(ocl_version_string, ' ');

    if (tokens.size() > 1) {
        auto version_string = tokens[1];
        auto version_tokens = split(version_string, '.');
        if (version_tokens.size() == 2) {
            ocl_major = std::stoi(version_tokens[0]);
            ocl_minor = std::stoi(version_tokens[1]);
        }
    }

    if (ocl_major != -1 && ocl_minor != -1) {
        int32_t ocl_version = ocl_major*100 + ocl_minor*10;
#if CL_TARGET_OPENCL_VERSION >= 200
        int32_t min_ocl_version = 200;
#else
        int32_t min_ocl_version = 120;
#endif
        if (ocl_version < min_ocl_version)
            return false;
    }

    return true;
}

// The priority return by this function impacts the order of devices reported by GPU plugin and devices enumeration
// Lower priority value means lower device ID
// Current behavior is: Intel iGPU < Intel dGPU < any other GPU
// Order of Intel dGPUs is undefined and depends on the OCL impl
// Order of other vendor GPUs is undefined and depends on the OCL impl
size_t get_device_priority(const cldnn::device_info& info) {
    if (info.vendor_id == cldnn::INTEL_VENDOR_ID && info.dev_type == cldnn::device_type::integrated_gpu) {
        return 0;
    } else if (info.vendor_id == cldnn::INTEL_VENDOR_ID) {
        return 1;
    } else {
        return std::numeric_limits<size_t>::max();
    }
}
}  // namespace

namespace cldnn {
namespace ocl {
static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
#ifdef _WIN32
static constexpr auto INTEL_D3D11_SHARING_EXT_NAME = "cl_khr_d3d11_sharing";
#endif // _WIN32

static std::vector<cl::Device> getSubDevices(cl::Device& rootDevice) {
    cl_uint maxSubDevices;
    size_t maxSubDevicesSize;
    const auto err = clGetDeviceInfo(rootDevice(),
                                     CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                                     sizeof(maxSubDevices),
                                     &maxSubDevices, &maxSubDevicesSize);

    OPENVINO_ASSERT(err == CL_SUCCESS && maxSubDevicesSize == sizeof(maxSubDevices),
                    "[GPU] clGetDeviceInfo(..., CL_DEVICE_PARTITION_MAX_SUB_DEVICES,...)");
    if (maxSubDevices == 0) {
        return {};
    }

    const auto partitionProperties = rootDevice.getInfo<CL_DEVICE_PARTITION_PROPERTIES>();
    const auto partitionable = std::any_of(partitionProperties.begin(), partitionProperties.end(),
                                            [](const decltype(partitionProperties)::value_type& prop) {
                                                return prop == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
                                            });

    if (!partitionable) {
        return {};
    }

    const auto partitionAffinityDomain = rootDevice.getInfo<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>();
    const decltype(partitionAffinityDomain) expectedFlags =
        CL_DEVICE_AFFINITY_DOMAIN_NUMA | CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;

    if ((partitionAffinityDomain & expectedFlags) != expectedFlags) {
        return {};
    }

    std::vector<cl::Device> subDevices;
    cl_device_partition_property partitionProperty[] = {CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                                                        CL_DEVICE_AFFINITY_DOMAIN_NUMA, 0};

    rootDevice.createSubDevices(partitionProperty, &subDevices);

    return subDevices;
}

std::vector<device::ptr> ocl_device_detector::sort_devices(const std::vector<device::ptr>& devices_list) {
    std::vector<device::ptr> sorted_list = devices_list;
    std::stable_sort(sorted_list.begin(), sorted_list.end(), [](device::ptr d1,  device::ptr d2) {
        return get_device_priority(d1->get_info()) < get_device_priority(d2->get_info());
    });

    return sorted_list;
}

std::map<std::string, device::ptr> ocl_device_detector::get_available_devices(void* user_context,
                                                                              void* user_device,
                                                                              int ctx_device_id,
                                                                              int target_tile_id) const {
    std::vector<device::ptr> devices_list;
    if (user_context != nullptr) {
        devices_list = create_device_list_from_user_context(user_context, ctx_device_id);
    } else if (user_device != nullptr) {
        devices_list = create_device_list_from_user_device(user_device);
    } else {
        devices_list = create_device_list();
    }

    devices_list = sort_devices(devices_list);

    std::map<std::string, device::ptr> ret;
    uint32_t idx = 0;
    for (auto& dptr : devices_list) {
        auto map_id = std::to_string(idx++);
        ret[map_id] = dptr;

        auto root_device = std::dynamic_pointer_cast<ocl_device>(dptr);
        OPENVINO_ASSERT(root_device != nullptr, "[GPU] Invalid device type created in ocl_device_detector");

        auto sub_devices = getSubDevices(root_device->get_device());
        if (!sub_devices.empty()) {
            uint32_t sub_idx = 0;
            for (auto& sub_device : sub_devices) {
                if (target_tile_id != -1 && static_cast<int>(sub_idx) != target_tile_id) {
                    sub_idx++;
                    continue;
                }
                auto sub_device_ptr = std::make_shared<ocl_device>(sub_device, cl::Context(sub_device), root_device->get_platform());
                ret[map_id + "." + std::to_string(sub_idx++)] = sub_device_ptr;
            }
        }
    }
    return ret;
}

std::vector<device::ptr> ocl_device_detector::create_device_list() const {
    cl_uint num_platforms = 0;
    // Get number of platforms availible
    cl_int error_code = clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0 || error_code == CL_PLATFORM_NOT_FOUND_KHR) {
        return {};
    }

    OPENVINO_ASSERT(error_code == CL_SUCCESS, create_device_error_msg, "[GPU] clGetPlatformIDs error code: ", std::to_string(error_code));
    // Get platform list
    std::vector<cl_platform_id> platform_ids(num_platforms);
    error_code = clGetPlatformIDs(num_platforms, platform_ids.data(), NULL);
    OPENVINO_ASSERT(error_code == CL_SUCCESS, create_device_error_msg, "[GPU] clGetPlatformIDs error code: ", std::to_string(error_code));

    std::vector<device::ptr> supported_devices;
    for (auto& id : platform_ids) {
        cl::Platform platform = cl::Platform(id);

        try {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (auto& device : devices) {
                if (!does_device_match_config(device))
                    continue;
                supported_devices.emplace_back(std::make_shared<ocl_device>(device, cl::Context(device), platform));
            }
        } catch (std::exception& ex) {
            GPU_DEBUG_LOG << "Devices query/creation failed for " << platform.getInfo<CL_PLATFORM_NAME>() << ": " << ex.what() << std::endl;
            GPU_DEBUG_LOG << "Platform is skipped" << std::endl;
            continue;
        }
    }
    return supported_devices;
}

std::vector<device::ptr> ocl_device_detector::create_device_list_from_user_context(void* user_context, int ctx_device_id) const {
    cl::Context ctx = cl::Context(static_cast<cl_context>(user_context), true);
    auto all_devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<device::ptr> supported_devices;
    for (size_t i = 0; i < all_devices.size(); i++) {
        auto& device = all_devices[i];
        if (!does_device_match_config(device) || static_cast<int>(i) != ctx_device_id)
            continue;
        supported_devices.emplace_back(std::make_shared<ocl_device>(device, ctx, cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>())));
    }

    OPENVINO_ASSERT(!supported_devices.empty(), "[GPU] User defined context does not have supported GPU device.");
    return supported_devices;
}

std::vector<device::ptr> ocl_device_detector::create_device_list_from_user_device(void* user_device) const {
    cl_uint num_platforms = 0;
    // Get number of platforms availible
    cl_int error_code = clGetPlatformIDs(0, NULL, &num_platforms);
    OPENVINO_ASSERT(error_code == CL_SUCCESS, create_device_error_msg, "[GPU] clGetPlatformIDs error code: ", std::to_string(error_code));

    // Get platform list
    std::vector<cl_platform_id> platform_ids(num_platforms);
    error_code = clGetPlatformIDs(num_platforms, platform_ids.data(), NULL);
    OPENVINO_ASSERT(error_code == CL_SUCCESS, create_device_error_msg, "[GPU] clGetPlatformIDs error code: ", std::to_string(error_code));

    std::vector<device::ptr> supported_devices;
    for (auto& id : platform_ids) {
        cl::PlatformVA platform = cl::PlatformVA(id);

        if (platform.getInfo<CL_PLATFORM_VENDOR>() != INTEL_PLATFORM_VENDOR)
            continue;

        std::vector<cl::Device> devices;
#ifdef _WIN32
        // From OpenCL Docs:
        // A non-NULL return value for clGetExtensionFunctionAddressForPlatform
        // does not guarantee that an extension function is actually supported
        // by the platform. The application must also make a corresponding query
        // using clGetPlatformInfo (platform, CL_PLATFORM_EXTENSIONS, …​ ) or
        // clGetDeviceInfo (device,CL_DEVICE_EXTENSIONS, …​ )
        // to determine if an extension is supported by the OpenCL implementation.
        const std::string& ext = platform.getInfo<CL_PLATFORM_EXTENSIONS>();
        if (ext.empty() || ext.find(INTEL_D3D11_SHARING_EXT_NAME) == std::string::npos) {
            continue;
        }

        platform.getDevices(CL_D3D11_DEVICE_KHR,
            user_device,
            CL_PREFERRED_DEVICES_FOR_D3D11_KHR,
#else
        platform.getDevices(CL_VA_API_DISPLAY_INTEL,
            user_device,
            CL_PREFERRED_DEVICES_FOR_VA_API_INTEL,
#endif
            &devices);

        for (auto& device : devices) {
            if (!does_device_match_config(device))
                continue;

            cl_context_properties props[] = {
#ifdef _WIN32
                CL_CONTEXT_D3D11_DEVICE_KHR,
#else
                CL_CONTEXT_VA_API_DISPLAY_INTEL,
#endif
                (intptr_t)user_device,
                CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                CL_CONTEXT_PLATFORM, (cl_context_properties)id,
                0 };
            supported_devices.emplace_back(std::make_shared<ocl_device>(device, cl::Context(device, props), platform));
        }
    }
    OPENVINO_ASSERT(!supported_devices.empty(), "[GPU] User specified device is not supported.");
    return supported_devices;
}

}  // namespace ocl
}  // namespace cldnn
