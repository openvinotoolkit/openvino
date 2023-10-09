// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/ov_plugin_cache.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <ie_plugin_config.hpp>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace utils {
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
    if (std::getenv("DISABLE_PLUGIN_CACHE") != nullptr) {
#ifndef NDEBUG
        std::cout << "'DISABLE_PLUGIN_CACHE' environment variable is set. New Core object will be created!"
                  << std::endl;
#endif
        return std::make_shared<ov::Core>();
    }
#ifndef NDEBUG
    std::cout << "Access PluginCache ov core. OV Core use count: " << ov_core.use_count() << std::endl;
#endif

    if (!ov_core) {
#ifndef NDEBUG
        std::cout << "Created ov core." << std::endl;
#endif
        ov_core = std::make_shared<ov::Core>();
    }
    assert(0 != ov_core.use_count());

    // register template plugin if it is needed
    try {
        std::string pluginName = "openvino_template_plugin";
        pluginName += OV_BUILD_POSTFIX;
        ov_core->register_plugin(
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(), pluginName),
            "TEMPLATE");
    } catch (...) {
    }

    if (!deviceToCheck.empty()) {
        auto properties = ov_core->get_property(deviceToCheck, ov::supported_properties);

        if (std::find(properties.begin(), properties.end(), ov::available_devices) != properties.end()) {
            auto availableDevices = ov_core->get_property(deviceToCheck, ov::available_devices);

            if (availableDevices.empty()) {
                std::cerr << "No available devices for " << deviceToCheck << std::endl;
                std::exit(EXIT_FAILURE);
            }

#ifndef NDEBUG
            std::cout << "Available devices for " << deviceToCheck << ":" << std::endl;

            for (const auto& device : availableDevices) {
                std::cout << "    " << device << std::endl;
            }
#endif
        }
    }
    return ov_core;
}

void PluginCache::reset() {
    std::lock_guard<std::mutex> lock(g_mtx);

#ifndef NDEBUG
    std::cout << "Reset PluginCache. OV Core use count: " << ov_core.use_count() << std::endl;
#endif

    ov_core.reset();
}

PluginCache::PluginCache() {
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestListener);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
