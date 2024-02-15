// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/ov_plugin_cache.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace utils {

ov::AnyMap pluginConfig = {};

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

    // Register Template plugin as a reference provider
    const auto devices = ov_core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), std::string(ov::test::utils::DEVICE_TEMPLATE)) == devices.end()) {
        auto plugin_path =
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                               std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
        if (!ov::util::file_exists(plugin_path)) {
            OPENVINO_THROW("Plugin: " + plugin_path + " does not exists!");
        }
        ov_core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
    }

    if (!target_device.empty()) {
        auto properties = ov_core.get_property(target_device, ov::supported_properties);

        if (std::find(properties.begin(), properties.end(), ov::available_devices) != properties.end()) {
            const auto available_devices = ov_core.get_property(target_device, ov::available_devices);
            if (available_devices.empty()) {
                OPENVINO_THROW("No available devices for " + target_device);
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

    if (!ov_core) {
        ov_core = std::make_shared<ov::Core>(create_core(deviceToCheck));
    }
    assert(0 != ov_core.use_count());
    return ov_core;
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
