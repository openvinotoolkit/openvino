// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "conformance.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace conformance {

inline const std::string getPluginLibNameByDevice(const std::string& deviceName) {
    const std::map<std::string, std::string> devices{
            { "AUTO", "openvino_auto_plugin" },
            { "HDDL", "openvino_intel_hddl_plugin" },
            { "VPUX", "openvino_intel_vpux_plugin" },
            { "CPU", "openvino_intel_cpu_plugin" },
            { "GNA", "openvino_intel_gna_plugin" },
            { "GPU", "openvino_intel_gpu_plugin" },
            { "HETERO", "openvino_hetero_plugin" },
            { "BATCH", "openvino_auto_batch_plugin" },
            { "MULTI", "openvino_auto_plugin" },
            { "MYRIAD", "openvino_intel_myriad_plugin" },
            { "TEMPLATE", "openvino_template_plugin" },
    };
    if (devices.find(deviceName) == devices.end()) {
        throw std::runtime_error("Incorrect device name");
    }
    return devices.at(deviceName);
}

inline const std::pair<std::string, std::string> generateDefaultMultiConfig() {
    return {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::conformance::targetDevice};
}

inline const std::pair<std::string, std::string> generateDefaultHeteroConfig() {
    return { "TARGET_FALLBACK" , ov::test::conformance::targetDevice };
}

inline const std::pair<std::string, std::string> generateDefaultBatchConfig() {
    // auto-batching with batch 1 (no real batching in fact, but full machinery is in action)
    return { CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(ov::test::conformance::targetDevice)};
}

inline const std::vector<std::map<std::string, std::string>> generateConfigs(const std::string& targetDevice,
                                                                             const std::vector<std::map<std::string, std::string>>& config = {}) {
    std::pair<std::string, std::string> defaultConfig;
    if (targetDevice ==  std::string(CommonTestUtils::DEVICE_MULTI) || targetDevice ==  std::string(CommonTestUtils::DEVICE_AUTO)) {
        defaultConfig = generateDefaultMultiConfig();
    } else if (targetDevice ==  std::string(CommonTestUtils::DEVICE_HETERO)) {
        defaultConfig = generateDefaultHeteroConfig();
    } else if (targetDevice ==  std::string(CommonTestUtils::DEVICE_BATCH)) {
        defaultConfig = generateDefaultBatchConfig();
    } else {
        throw std::runtime_error("Incorrect target device: " + targetDevice);
    }

    std::vector<std::map<std::string, std::string>> resultConfig;
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
    return deviceName + ":" + ov::test::conformance::targetDevice;
}

inline const std::vector<std::string> returnAllPossibleDeviceCombination() {
    std::vector<std::string> res{ov::test::conformance::targetDevice};
    std::vector<std::string> devices{CommonTestUtils::DEVICE_HETERO, CommonTestUtils::DEVICE_AUTO,
                                     CommonTestUtils::DEVICE_BATCH, CommonTestUtils::DEVICE_MULTI};
    for (const auto& device : devices) {
        res.emplace_back(generateComplexDeviceName(device));
    }
    return res;
}

const std::vector<std::map<std::string, std::string>> emptyConfig = {
        {},
};

}  // namespace conformance
}  // namespace test
}  // namespace ov
