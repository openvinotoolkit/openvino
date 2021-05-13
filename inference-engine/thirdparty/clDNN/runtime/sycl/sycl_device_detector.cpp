// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_device_detector.hpp"
#include "sycl_device.hpp"

#include <string>
#include <vector>
#include <list>
#include <utility>

namespace cldnn {
namespace sycl {

std::map<std::string, device::ptr> sycl_device_detector::get_available_devices(runtime_types runtime_type, void* user_context, void* user_device) const {
    std::vector<device::ptr> dev_orig, dev_sorted;
    if (user_context != nullptr) {
        dev_orig = create_device_list_from_user_context(user_context);
    } else if (user_device != nullptr) {
        dev_orig = create_device_list_from_user_device(user_device);
    } else {
        dev_orig = create_device_list(runtime_type);
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
        ret[std::to_string(idx++)] = dptr;
    }
    return ret;
}

std::vector<device::ptr> sycl_device_detector::create_device_list(runtime_types runtime_type) const {
    std::vector<cl::sycl::device> devices;
    auto platforms = cl::sycl::platform::get_platforms();

    for (const auto &p : platforms) {
        auto p_devices = p.get_devices(dev_type);
        devices.insert(devices.end(), p_devices.begin(), p_devices.end());
    }

    devices.erase(std::remove_if(devices.begin(), devices.end(),
        [=](const cl::sycl::device &dev) {
            auto _vendor_id = dev.get_info<cl::sycl::info::device::vendor_id>();
            if (_vendor_id != dev_vendor)
                return true;

            auto _dev_type = dev.get_info<cl::sycl::info::device::device_type>();
            if (_dev_type != dev_type)
                return true;

            auto platform_name = dev.get_platform().get_info<cl::sycl::info::platform::name>();
            std::string expected_platrform_name = "";
            if (runtime_type == runtime_types::ocl) {
                expected_platrform_name = "OpenCL";
            } else {
                expected_platrform_name = "Level";
            }

            if (platform_name.find(expected_platrform_name) == std::string::npos) {
                return true;
            }

            return false;
        }),
        devices.end());

    std::vector<device::ptr> ret;
    for (auto& dev : devices) {
        auto ctx = cl::sycl::context(dev);
        ret.emplace_back(std::make_shared<sycl_device>(dev, ctx));
    }
    return ret;
}

std::vector<device::ptr> sycl_device_detector::create_device_list_from_user_context(void* user_context) const {
    throw std::runtime_error("create_device_list_from_user_context is not implemented yet for sycl backend");
}

std::vector<device::ptr> sycl_device_detector::create_device_list_from_user_device(void* user_device) const {
    throw std::runtime_error("create_device_list_from_user_device is not implemented yet for sycl backend");
}

bool sycl_device_detector::does_device_match_config(const cl::sycl::device& device) const {
    // Is it intel gpu
    if (device.get_info<cl::sycl::info::device::device_type>() != dev_type ||
        device.get_info<cl::sycl::info::device::vendor_id>() != dev_vendor) {
        return false;
    }

    return true;
}

}  // namespace sycl
}  // namespace cldnn
