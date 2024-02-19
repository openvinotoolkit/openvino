// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_plugin_cache.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace utils {

ov::AnyMap pluginConfig = {};

void register_template_plugin(ov::Core& ov_core) {
    auto plugin_path =
        ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                           std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
    if (!ov::util::file_exists(plugin_path)) {
        OPENVINO_THROW("Plugin: " + plugin_path + " does not exists!");
    }
    ov_core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
}

ov::Core create_core(const std::string& target_device) {
    ov::Core ov_core;

    if (!pluginConfig.empty()) {
        if (!target_device.empty()) {
            auto properties = ov_core.get_property(target_device, ov::supported_properties);
            for (auto& property : pluginConfig) {
                if (std::find(properties.begin(), properties.end(), property.first) == properties.end()) {
                    OPENVINO_THROW("Property " + property.first +
                                   ", which was tryed to set in --config file, is not supported by " + target_device);
                }
            }
            ov_core.set_property(target_device, pluginConfig);
        } else {
            ov_core.set_property(pluginConfig);
        }
    }

    auto core_registered_devices = PluginCache::get().get_core_registered_devices();
    if (core_registered_devices.empty()) {
        PluginCache::get().set_core_registered_devices(ov_core.get_available_devices());
    }

    // Register Template plugin as a reference provider
    if (std::find(core_registered_devices.begin(),
                  core_registered_devices.end(),
                  std::string(ov::test::utils::DEVICE_TEMPLATE)) == core_registered_devices.end()) {
        register_template_plugin(ov_core);
        core_registered_devices.push_back(ov::test::utils::DEVICE_TEMPLATE);
    }

    if (!target_device.empty()) {
        auto properties = ov_core.get_property(target_device, ov::supported_properties);

        if (std::find(properties.begin(), properties.end(), ov::available_devices) != properties.end()) {
            if (std::find_if(core_registered_devices.begin(),
                             core_registered_devices.end(),
                             [](const std::string& device) {
                                 return device.find(std::string(ov::test::utils::DEVICE_TEMPLATE)) != std::string::npos;
                             }) == core_registered_devices.end())
                OPENVINO_THROW("No available devices for " + target_device);

#ifndef NDEBUG
            std::cout << "Available devices :" << std::endl;
            for (const auto& device : core_registered_devices) {
                std::cout << "    " << device << std::endl;
            }
#endif
        }
    }

    return ov_core;
}

namespace {
class TestListener : public testing::EmptyTestEventListener {
public:
    void OnTestEnd(const testing::TestInfo& testInfo) override {
        if (auto testResult = testInfo.result()) {
            if (testResult->Failed()) {
                PluginCache::get().reset();
            }
        }
    }
};
}  // namespace

PluginCache& PluginCache::get() {
    static PluginCache instance;
    return instance;
}

std::shared_ptr<ov::Core> PluginCache::core(const std::string& deviceToCheck) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (disable_plugin_cache) {
        return std::make_shared<ov::Core>(create_core(deviceToCheck));
    }

#ifndef NDEBUG
    std::cout << "Access PluginCache ov core. OV Core use count: " << ov_core.use_count() << std::endl;
#endif

    if (!ov_core) {
#ifndef NDEBUG
        std::cout << "Created ov core." << std::endl;
#endif
        ov_core = std::make_shared<ov::Core>(create_core(deviceToCheck));
        assert(0 != ov_core.use_count());
    }

    return ov_core;
}

std::vector<std::string> PluginCache::get_core_registered_devices() {
    return core_registered_devices;
}

void PluginCache::set_core_registered_devices(std::vector<std::string> devices) {
    for (auto& device : devices) {
        if (std::find(core_registered_devices.begin(), core_registered_devices.end(), device) ==
            core_registered_devices.end()) {
            core_registered_devices.push_back(device);
        }
    }
}

void PluginCache::reset() {
    std::lock_guard<std::mutex> lock(g_mtx);
    ov_core.reset();
}

PluginCache::PluginCache() {
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestListener);
    disable_plugin_cache = std::getenv("DISABLE_PLUGIN_CACHE") == nullptr ? false : true;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
