// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include <algorithm>
#include <stdexcept>

#include <functional_test_utils/ov_plugin_cache.hpp>

#include "set_device_name.hpp"

namespace ov {
namespace test {

void set_device_suffix(const std::string& suffix) {
    std::string new_gpu_name = CommonTestUtils::DEVICE_GPU;
    new_gpu_name += CommonTestUtils::DEVICE_SUFFIX_SEPARATOR;
    new_gpu_name += suffix;
    auto available_devices = utils::PluginCache::get().core()->get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), new_gpu_name) == available_devices.end()) {
        throw std::runtime_error("The device " + new_gpu_name + " is not in the available devices! Please use other on!");
    }
    CommonTestUtils::DEVICE_GPU = new_gpu_name.c_str();
}

} // namespace test
} // namespace ov