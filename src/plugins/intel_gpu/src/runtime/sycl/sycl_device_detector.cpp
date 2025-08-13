// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_device_detector.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "sycl_device.hpp"
#include "sycl_common.hpp"

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
    "[GPU] No supported SYCL devices found or unexpected error happened during devices query.\n"
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

bool does_device_match_config(const ::sycl::device& device) {
    // TODO: change behavior when commit
    if (device.is_gpu()) {
        return false;
    }

    int32_t sycl_major = -1;
    int32_t sycl_minor = -1;
    // Spec says that the format of this string is OpenCL<space><major_version.minor_version><space><vendor-specific information>
    auto sycl_version_string = device.get_info<::sycl::info::device::version>();
    std::cout << "SYCL version string: " << sycl_version_string << std::endl;
    auto tokens = split(sycl_version_string, ' ');

    if (tokens.size() > 1) {
        auto version_string = tokens[1];
        auto version_tokens = split(version_string, '.');
        if (version_tokens.size() == 2) {
            sycl_major = std::stoi(version_tokens[0]);
            sycl_minor = std::stoi(version_tokens[1]);
        }
    }

    if (sycl_major != -1 && sycl_minor != -1) {
        int32_t sycl_version = sycl_major*100 + sycl_minor*10;
#if CL_TARGET_OPENCL_VERSION >= 200
        int32_t min_sycl_version = 200;
#else
        int32_t min_sycl_version = 120;
#endif
        if (sycl_version < min_sycl_version)
            return false;
    }

    return true;
}

// The priority return by this function impacts the order of devices reported by GPU plugin and devices enumeration
// Lower priority value means lower device ID
// Current behavior is: Intel iGPU < Intel dGPU < any other GPU
// Order of Intel dGPUs is undefined and depends on the SYCL impl
// Order of other vendor GPUs is undefined and depends on the SYCL impl
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
namespace sycl {
static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
#ifdef _WIN32
static constexpr auto INTEL_D3D11_SHARING_EXT_NAME = "cl_khr_d3d11_sharing";
#endif // _WIN32

static std::vector<::sycl::device> getSubDevices(::sycl::device& rootDevice) {
    uint32_t maxSubDevices = rootDevice.get_info<::sycl::info::device::partition_max_sub_devices>();
    if (maxSubDevices == 0) {
        return {};
    }

    const auto partitionProperties = rootDevice.get_info<::sycl::info::device::partition_properties>();
    const auto partitionable = std::any_of(partitionProperties.begin(), partitionProperties.end(),
                                            [](const decltype(partitionProperties)::value_type& prop) {
                                                return prop == ::sycl::info::partition_property::partition_by_affinity_domain;
                                            });

    if (!partitionable) {
        return {};
    }

    const auto partitionAffinityDomains = rootDevice.get_info<::sycl::info::device::partition_affinity_domains>();
    if (std::find(partitionAffinityDomains.begin(), partitionAffinityDomains.end(),
                  ::sycl::info::partition_affinity_domain::numa) == partitionAffinityDomains.end() ||
        std::find(partitionAffinityDomains.begin(), partitionAffinityDomains.end(),
                  ::sycl::info::partition_affinity_domain::next_partitionable) == partitionAffinityDomains.end()) {
        return {};
    }

    auto subDevices = rootDevice.create_sub_devices<::sycl::info::partition_property::partition_by_affinity_domain>
                                                                        (::sycl::info::partition_affinity_domain::numa);

    return subDevices;
}

std::vector<device::ptr> sycl_device_detector::sort_devices(const std::vector<device::ptr>& devices_list) {
    std::vector<device::ptr> sorted_list = devices_list;
    std::stable_sort(sorted_list.begin(), sorted_list.end(), [](device::ptr d1,  device::ptr d2) {
        return get_device_priority(d1->get_info()) < get_device_priority(d2->get_info());
    });

    return sorted_list;
}

std::map<std::string, device::ptr> sycl_device_detector::get_available_devices(void* user_context,
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

        auto root_device = std::dynamic_pointer_cast<sycl_device>(dptr);
        OPENVINO_ASSERT(root_device != nullptr, "[GPU] Invalid device type created in sycl_device_detector");

        auto sub_devices = getSubDevices(root_device->get_device());
        if (!sub_devices.empty()) {
            uint32_t sub_idx = 0;
            for (auto& sub_device : sub_devices) {
                if (target_tile_id != -1 && static_cast<int>(sub_idx) != target_tile_id) {
                    sub_idx++;
                    continue;
                }
                auto sub_device_ptr = std::make_shared<sycl_device>(sub_device, ::sycl::context(sub_device), root_device->get_platform());
                ret[map_id + "." + std::to_string(sub_idx++)] = sub_device_ptr;
            }
        }
    }
    return ret;
}

std::vector<device::ptr> sycl_device_detector::create_device_list() const {
    auto platforms = ::sycl::platform::get_platforms();

    std::vector<device::ptr> supported_devices;
    for (auto& platform : platforms) {
        try {
            auto devices = platform.get_devices();
            for (auto& device : devices) {
                if (!does_device_match_config(device))
                    continue;
                supported_devices.emplace_back(std::make_shared<sycl_device>(device, ::sycl::context(device), platform));
            }
        } catch (std::exception& ex) {
            GPU_DEBUG_LOG << "Devices query/creation failed for " << platform.get_info<::sycl::info::platform::name>() << ": " << ex.what() << std::endl;
            GPU_DEBUG_LOG << "Platform is skipped" << std::endl;
            continue;
        }
    }
    return supported_devices;
}

std::vector<device::ptr> sycl_device_detector::create_device_list_from_user_context(void* user_context, int ctx_device_id) const {
    auto& ctx = *(static_cast<::sycl::context*>(user_context));
    auto all_devices = ctx.get_devices();

    std::vector<device::ptr> supported_devices;
    for (size_t i = 0; i < all_devices.size(); i++) {
        auto& device = all_devices[i];
        if (!does_device_match_config(device) || static_cast<int>(i) != ctx_device_id)
            continue;
        supported_devices.emplace_back(std::make_shared<sycl_device>(device, ctx, device.get_platform()));
    }

    OPENVINO_ASSERT(!supported_devices.empty(), "[GPU] User defined context does not have supported GPU device.");
    return supported_devices;
}

std::vector<device::ptr> sycl_device_detector::create_device_list_from_user_device(void* user_device) const {
    OPENVINO_NOT_IMPLEMENTED;
    // TODO: implement
}

}  // namespace sycl
}  // namespace cldnn
