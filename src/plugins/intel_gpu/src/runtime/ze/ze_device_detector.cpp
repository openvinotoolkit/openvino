// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_device_detector.hpp"
#include "ze_device.hpp"
#include "ze_common.hpp"

#include <string>
#include <vector>
#include <list>
#include <utility>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace {
// bool does_device_match_config(bool out_of_order, const cl::Device& device) {
// // Is it intel gpu
// if (device.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_GPU ||
//     device.getInfo<CL_DEVICE_VENDOR_ID>() != 0x8086) {
//     return false;
// }

// // Does device support OOOQ?
// if (out_of_order) {
//     auto queue_properties = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
//     using cmp_t = std::common_type<decltype(queue_properties),
//         typename std::underlying_type<cl::QueueProperties>::type>::type;
//     if (!(static_cast<cmp_t>(queue_properties) & static_cast<cmp_t>(cl::QueueProperties::OutOfOrder))) {
//         return false;
//     }
// }

// return true;
// }
}  // namespace
namespace cldnn {
namespace ze {
// static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";

// static std::vector<cl::Device> getSubDevices(cl::Device& rootDevice) {
//     cl_uint maxSubDevices;
//     size_t maxSubDevicesSize;
//     const auto err = clGetDeviceInfo(rootDevice(),
//                                      CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
//                                      sizeof(maxSubDevices),
//                                      &maxSubDevices, &maxSubDevicesSize);

//     if (err != CL_SUCCESS || maxSubDevicesSize != sizeof(maxSubDevices)) {
//         throw cl::Error(err, "clGetDeviceInfo(..., CL_DEVICE_PARTITION_MAX_SUB_DEVICES,...)");
//     }

//     if (maxSubDevices == 0) {
//         return {};
//     }

//     const auto partitionProperties = rootDevice.getInfo<CL_DEVICE_PARTITION_PROPERTIES>();
//     const auto partitionable = std::any_of(partitionProperties.begin(), partitionProperties.end(),
//                                             [](const decltype(partitionProperties)::value_type& prop) {
//                                                 return prop == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
//                                             });

//     if (!partitionable) {
//         return {};
//     }

//     const auto partitionAffinityDomain = rootDevice.getInfo<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>();
//     const decltype(partitionAffinityDomain) expectedFlags =
//         CL_DEVICE_AFFINITY_DOMAIN_NUMA | CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;

//     if ((partitionAffinityDomain & expectedFlags) != expectedFlags) {
//         return {};
//     }

//     std::vector<cl::Device> subDevices;
//     cl_device_partition_property partitionProperty[] = {CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
//                                                         CL_DEVICE_AFFINITY_DOMAIN_NUMA, 0};

//     rootDevice.createSubDevices(partitionProperty, &subDevices);

//     return subDevices;
// }

std::map<std::string, device::ptr> ze_device_detector::get_available_devices(void* user_context, void* user_device) const {
    bool host_out_of_order = true;  // Change to false, if debug requires in-order queue.
    std::vector<device::ptr> dev_orig, dev_sorted;
    if (user_context != nullptr) {
        dev_orig = create_device_list_from_user_context(host_out_of_order, user_context);
    } else if (user_device != nullptr) {
        dev_orig = create_device_list_from_user_device(host_out_of_order, user_device);
    } else {
        dev_orig = create_device_list(host_out_of_order);
    }

    std::map<std::string, device::ptr> ret;
    for (auto& dptr : dev_orig) {
        if (dptr->get_info().dev_type == cldnn::device_type::integrated_gpu)
            dev_sorted.insert(dev_sorted.begin(), dptr);
        else
            dev_sorted.push_back(dptr);
    }
    uint32_t idx = 0;
    for (auto& dptr : dev_sorted) {
        auto map_id = std::to_string(idx++);
        ret[map_id] = dptr;

        // auto rootDevice = std::dynamic_pointer_cast<ze_device>(dptr);
        // if (!rootDevice) {
        //     throw std::runtime_error("Invalid device type created in ze_device_detector");
        // }

        // auto subDevices = getSubDevices(rootDevice->get_device());
        // if (!subDevices.empty()) {
        //     uint32_t sub_idx = 0;
        //     for (auto& subdevice : subDevices) {
        //         auto subdPtr = std::make_shared<ze_device>(subdevice, cl::Context(subdevice), rootDevice->get_platform());
        //         ret[map_id+"."+std::to_string(sub_idx++)] = subdPtr;
        //     }
        // }
    }
    return ret;
}

std::vector<device::ptr> ze_device_detector::create_device_list(bool out_out_order) const {
    std::vector<device::ptr> ret;

    ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Discover all the driver instances
    uint32_t driver_count = 0;
    ZE_CHECK(zeDriverGet(&driver_count, nullptr));

    std::vector<ze_driver_handle_t> all_drivers(driver_count);
    ZE_CHECK(zeDriverGet(&driver_count, &all_drivers[0]));

    // Find a driver instance with a GPU device
    for (uint32_t i = 0; i < driver_count; ++i) {
        uint32_t device_count = 0;
        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, nullptr));

        std::vector<ze_device_handle_t> all_devices(device_count);
        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, &all_devices[0]));

        for (uint32_t d = 0; d < device_count; ++d) {
            ze_device_properties_t device_properties;
            ZE_CHECK(zeDeviceGetProperties(all_devices[d], &device_properties));

            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                ret.emplace_back(std::make_shared<ze_device>(all_drivers[i], all_devices[d]));
            }
        }
    }

    if (ret.empty()) {
        throw std::runtime_error("[CLDNN ERROR]. No GPU device was found.");
    }

    return ret;
}

std::vector<device::ptr>  ze_device_detector::create_device_list_from_user_context(bool out_out_order, void* user_context) const {
    throw std::runtime_error("Not implemented");
}

std::vector<device::ptr>  ze_device_detector::create_device_list_from_user_device(bool out_out_order, void* user_device) const {
    throw std::runtime_error("Not implemented");
}

}  // namespace ze
}  // namespace cldnn
