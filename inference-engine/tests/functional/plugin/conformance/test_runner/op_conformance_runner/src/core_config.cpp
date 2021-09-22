// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"

#include "conformance.hpp"

void CoreConfiguration(LayerTestsUtils::LayerTestsCommon* test) {
    std::shared_ptr<InferenceEngine::Core> core = PluginCache::get().ie();
    auto availableDevices = core->GetAvailableDevices();
    std::string targetDevice = std::string(ConformanceTests::targetDevice);
    if (std::find(availableDevices.begin(), availableDevices.end(), targetDevice) == availableDevices.end()) {
        core->RegisterPlugin(ConformanceTests::targetPluginName,
                             ConformanceTests::targetDevice);
    }
}
