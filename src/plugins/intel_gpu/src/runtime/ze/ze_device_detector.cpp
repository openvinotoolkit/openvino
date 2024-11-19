// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_device_detector.hpp"
#include "ze_device.hpp"
#include "ze_common.hpp"
#include <ze_api.h>
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/except.hpp"

#include <vector>

namespace cldnn {
namespace ze {

static std::vector<ze_device_handle_t> get_sub_devices(ze_device_handle_t root_device) {
    uint32_t n_subdevices = 0;
    ZE_CHECK(zeDeviceGetSubDevices(root_device, &n_subdevices, nullptr));
    if (n_subdevices == 0)
        return {};

    std::vector<ze_device_handle_t> subdevices(n_subdevices);

    ZE_CHECK(zeDeviceGetSubDevices(root_device, &n_subdevices, &subdevices[0]));

    return subdevices;
}

std::map<std::string, device::ptr> ze_device_detector::get_available_devices(void* user_context,
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

        auto root_device = std::dynamic_pointer_cast<ze_device>(dptr);
        OPENVINO_ASSERT(root_device != nullptr, "[GPU] Invalid device type created in ocl_device_detector");

        auto sub_devices = get_sub_devices(root_device->get_device());
        if (!sub_devices.empty()) {
            uint32_t sub_idx = 0;
            for (auto& sub_device : sub_devices) {
                if (target_tile_id != -1 && static_cast<int>(sub_idx) != target_tile_id) {
                    sub_idx++;
                    continue;
                }
                auto sub_device_ptr = std::make_shared<ze_device>(root_device->get_driver(), sub_device);
                ret[map_id + "." + std::to_string(sub_idx++)] = sub_device_ptr;
            }
        }
    }

    return ret;
}

std::vector<device::ptr> ze_device_detector::create_device_list() const {
    std::vector<device::ptr> ret;

    ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count = 0;
    ZE_CHECK(zeDriverGet(&driver_count, nullptr));

    std::vector<ze_driver_handle_t> all_drivers(driver_count);
    ZE_CHECK(zeDriverGet(&driver_count, &all_drivers[0]));

    for (uint32_t i = 0; i < driver_count; ++i) {
        uint32_t device_count = 0;
        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, nullptr));

        std::vector<ze_device_handle_t> all_devices(device_count);
        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, &all_devices[0]));

        for (uint32_t d = 0; d < device_count; ++d) {
            try {
                ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
                ZE_CHECK(zeDeviceGetProperties(all_devices[d], &device_properties));

                if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                    ret.emplace_back(std::make_shared<ze_device>(all_drivers[i], all_devices[d]));
                }
            } catch (std::exception& ex) {
                GPU_DEBUG_LOG << "Devices query/creation failed for driver " << i << ex.what() << std::endl;
                GPU_DEBUG_LOG << "Platform is skipped" << std::endl;
                continue;
            }
        }
    }

    return ret;
}

std::vector<device::ptr> ze_device_detector::create_device_list_from_user_context(void* user_context, int ctx_device_id) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<device::ptr> ze_device_detector::create_device_list_from_user_device(void* user_device) const {
    OPENVINO_NOT_IMPLEMENTED;
}

}  // namespace ze
}  // namespace cldnn
