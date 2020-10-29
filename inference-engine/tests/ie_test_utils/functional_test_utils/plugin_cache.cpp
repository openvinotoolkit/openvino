// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_cache.hpp"

#include <cstdlib>

#include <unordered_map>
#include <mutex>

#include <gtest/gtest.h>
#include <ie_plugin_config.hpp>

namespace {

std::shared_ptr<InferenceEngine::Core> ie_core;
std::mutex g_mtx;

class TestListener : public testing::EmptyTestEventListener {
public:
    void OnTestEnd(const testing::TestInfo &testInfo) override {
        if (auto testResult = testInfo.result()) {
            if (testResult->Failed()) {
                PluginCache::get().reset();
            }
        }
    }
};

}  // namespace

PluginCache &PluginCache::get() {
    static PluginCache instance;
    return instance;
}

std::shared_ptr<InferenceEngine::Core> PluginCache::ie(const std::string &deviceToCheck) const {
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

    if (!deviceToCheck.empty()) {
        std::vector<std::string> metrics = ie_core->GetMetric(deviceToCheck, METRIC_KEY(SUPPORTED_METRICS));

        if (std::find(metrics.begin(), metrics.end(), METRIC_KEY(AVAILABLE_DEVICES)) != metrics.end()) {
            std::vector<std::string> availableDevices = ie_core->GetMetric(deviceToCheck,
                                                                           METRIC_KEY(AVAILABLE_DEVICES));

            if (availableDevices.empty()) {
                std::cerr << "No available devices for " << deviceToCheck << std::endl;
                std::exit(EXIT_FAILURE);
            }

            std::cout << "Available devices for " << deviceToCheck << ":" << std::endl;

            for (const auto &device : availableDevices) {
                std::cout << "    " << device << std::endl;
            }
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
    auto &listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestListener);
}
