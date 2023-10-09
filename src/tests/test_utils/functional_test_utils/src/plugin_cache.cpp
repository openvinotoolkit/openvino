// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/plugin_cache.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <ie_plugin_config.hpp>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "openvino/util/file_util.hpp"

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

std::shared_ptr<InferenceEngine::Core> PluginCache::ie(const std::string& deviceToCheck) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (std::getenv("DISABLE_PLUGIN_CACHE") != nullptr) {
#ifndef NDEBUG
        std::cout << "'DISABLE_PLUGIN_CACHE' environment variable is set. New Core object will be created!"
                  << std::endl;
#endif
        return std::make_shared<InferenceEngine::Core>();
    }
#ifndef NDEBUG
    std::cout << "Access PluginCache ie core. IE Core use count: " << ie_core.use_count() << std::endl;
#endif

    if (!ie_core) {
#ifndef NDEBUG
        std::cout << "Created ie core." << std::endl;
#endif
        ie_core = std::make_shared<InferenceEngine::Core>();
    }
    assert(0 != ie_core.use_count());

    // register template plugin if it is needed
    try {
        std::string pluginName = "openvino_template_plugin";
        pluginName += OV_BUILD_POSTFIX;
        ie_core->RegisterPlugin(
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(), pluginName),
            "TEMPLATE");
    } catch (...) {
    }

    if (!deviceToCheck.empty()) {
        std::vector<std::string> metrics;
        if (deviceToCheck.find(':') != std::string::npos) {
            std::string realDevice = deviceToCheck.substr(0, deviceToCheck.find(':'));
            metrics = {ie_core->GetMetric(realDevice, METRIC_KEY(SUPPORTED_METRICS)).as<std::string>()};
        } else {
            metrics = {ie_core->GetMetric(deviceToCheck, METRIC_KEY(SUPPORTED_METRICS)).as<std::string>()};
        }
        if (std::find(metrics.begin(), metrics.end(), METRIC_KEY(AVAILABLE_DEVICES)) != metrics.end()) {
            auto availableDevices =
                ie_core->GetMetric(deviceToCheck, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();

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
    return ie_core;
}

void PluginCache::reset() {
    std::lock_guard<std::mutex> lock(g_mtx);

#ifndef NDEBUG
    std::cout << "Reset PluginCache. IE Core use count: " << ie_core.use_count() << std::endl;
#endif

    ie_core.reset();
}

PluginCache::PluginCache() {
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestListener);
}
