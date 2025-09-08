// Copyright (C) 2018-2025 Intel Corporation
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

ov::AnyMap global_plugin_config = {};
std::unordered_set<std::string> available_devices = {};
std::string target_device = "";
std::string target_plugin_name = "";

void register_plugin(ov::Core& ov_core) noexcept {
    if (!target_plugin_name.empty()) {
        ov_core.register_plugin(target_plugin_name, target_device);
    }
}

void register_template_plugin(ov::Core& ov_core) noexcept {
    auto plugin_path =
        ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                           std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
    if (!ov::util::file_exists(plugin_path)) {
        OPENVINO_THROW("Plugin: " + plugin_path + " does not exists!");
    }
    ov_core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
}

ov::Core create_core(const std::string& in_target_device) {
    ov::Core ov_core;

#if !defined(OPENVINO_STATIC_LIBRARY) && !defined(USE_STATIC_IE)
    register_plugin(ov_core);
    // Register Template plugin as a reference provider
    register_template_plugin(ov_core);
#endif  // !OPENVINO_STATIC_LIBRARY && !USE_STATIC_IE

    if (available_devices.empty()) {
        const auto core_devices = ov_core.get_available_devices();
        available_devices.insert(core_devices.begin(), core_devices.end());
    }

    if (!available_devices.count(in_target_device) && !in_target_device.empty()) {
#ifndef NDEBUG
        std::cout << "Available devices :" << std::endl;
        for (const auto& device : available_devices) {
            std::cout << "    " << device << std::endl;
        }
#endif
        OPENVINO_THROW("No available devices for " + in_target_device);
    }

    if (!global_plugin_config.empty()) {
        // apply config to main device specified by user at launch or to special device specified when creating new сore
        auto config_device = in_target_device.empty() ? target_device : in_target_device;
        for (auto& property : global_plugin_config) {
            try {
                ov_core.set_property(config_device, global_plugin_config);
            } catch (...) {
                OPENVINO_THROW("Property " + property.first +
                               ", which was tryed to set in --config file, is not supported by " + target_device);
            }
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

std::shared_ptr<ov::Core> PluginCache::core(const std::string& target_device) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (disable_plugin_cache) {
        return std::make_shared<ov::Core>(create_core(target_device));
    }
    if (!ov_core) {
        ov_core = std::make_shared<ov::Core>(create_core(target_device));
        assert(0 != ov_core.use_count());
    }
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
