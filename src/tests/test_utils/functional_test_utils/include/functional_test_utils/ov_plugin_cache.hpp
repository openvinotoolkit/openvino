// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace utils {

static ov::Core create_core(const std::string& target_device = "") {
    ov::Core ov_core;
    // Register Template plugin as a reference provider
    const auto devices = ov_core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), std::string(ov::test::utils::DEVICE_TEMPLATE)) == devices.end()) {
        auto plugin_path =
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                               std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
        if (!ov::util::file_exists(plugin_path)) {
            throw std::runtime_error("Plugin: " + plugin_path + " does not exists!");
        }
        ov_core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
    }

    if (!target_device.empty()) {
        auto properties = ov_core.get_property(target_device, ov::supported_properties);

        if (std::find(properties.begin(), properties.end(), ov::available_devices) != properties.end()) {
            const auto available_devices = ov_core.get_property(target_device, ov::available_devices);
            if (available_devices.empty()) {
                std::string msg = "No available devices for " + target_device;
                throw std::runtime_error(msg);
            }
#ifndef NDEBUG
            std::cout << "Available devices :" << std::endl;
            for (const auto& device : available_devices) {
                std::cout << "    " << device << std::endl;
            }
#endif
        }
    }
    return ov_core;
}

class PluginCache {
public:
    std::shared_ptr<ov::Core> core(const std::string& deviceToCheck = std::string());

    static PluginCache& get();

    void reset();

    PluginCache(const PluginCache&) = delete;

    PluginCache& operator=(const PluginCache&) = delete;

private:
    PluginCache();

    ~PluginCache() = default;

    std::mutex g_mtx;
    std::shared_ptr<ov::Core> ov_core;
};
}  // namespace utils
}  // namespace test
}  // namespace ov
