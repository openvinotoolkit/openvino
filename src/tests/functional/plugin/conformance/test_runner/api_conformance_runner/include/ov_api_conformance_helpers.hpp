// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "conformance.hpp"
#include "common_test_utils/test_constants.hpp"

// TODO: fix namespaces
using namespace ConformanceTests;

namespace ov {
namespace test {
namespace conformance {

inline const std::string getPluginLibNameByDevice(const std::string& deviceName) {
    const std::map<std::string, std::string> devices{
            { "AUTO", "AutoPlugin" },
            { "HDDL", "HDDLPlugin" },
            { "VPUX", "VPUXPlugin" },
            { "AUTO", "ov_auto_plugin" },
            { "CPU", "MKLDNNPlugin" },
            { "GNA", "ov_intel_gna_plugin" },
            { "GPU", "ov_intel_gpu_plugin" },
            { "HETERO", "ov_hetero_plugin" },
            { "MULTI", "ov_multi_plugin" },
            { "MYRIAD", "myriadPlugin" },
            { "TEMPLATE", "templatePlugin" },
    };
    if (devices.find(deviceName) == devices.end()) {
        throw std::runtime_error("Incorrect device name");
    }
    return devices.at(deviceName);
}

inline const std::pair<std::string, ov::Any> generateDefaultMultiConfig() {
    return {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ConformanceTests::targetDevice};
}

inline const std::pair<std::string, ov::Any> generateDefaultHeteroConfig() {
    return { "TARGET_FALLBACK" , ConformanceTests::targetDevice };
}

inline const std::vector<ov::AnyMap> generateConfigs(const std::string& targetDevice,
                                                                         const std::vector<ov::AnyMap>& config = {}) {
    std::pair<std::string, ov::Any> defaultConfig;
    if (targetDevice ==  std::string(CommonTestUtils::DEVICE_MULTI) || targetDevice ==  std::string(CommonTestUtils::DEVICE_AUTO)) {
        defaultConfig = generateDefaultMultiConfig();
    } else if (targetDevice ==  std::string(CommonTestUtils::DEVICE_HETERO)) {
        defaultConfig = generateDefaultHeteroConfig();
    } else {
        throw std::runtime_error("Incorrect target device: " + targetDevice);
    }

    std::vector<ov::AnyMap> resultConfig;
    if (config.empty()) {
        return {{defaultConfig}};
    }
    for (auto configItem : config) {
        configItem.insert(defaultConfig);
        resultConfig.push_back(configItem);
    }
    return resultConfig;
}

inline const std::string generateComplexDeviceName(const std::string& deviceName) {
    return deviceName + ":" + ConformanceTests::targetDevice;
}

inline const std::vector<std::string> returnAllPossibleDeviceCombination() {
    std::vector<std::string> res{ConformanceTests::targetDevice};
    std::vector<std::string> devices{CommonTestUtils::DEVICE_HETERO, CommonTestUtils::DEVICE_AUTO, CommonTestUtils::DEVICE_MULTI};
    for (const auto& device : devices) {
        res.emplace_back(generateComplexDeviceName(device));
    }
    return res;
}

const std::vector<ov::AnyMap> emptyConfig = {
        {},
};

}  // namespace conformance
}  // namespace test
}  // namespace ov
