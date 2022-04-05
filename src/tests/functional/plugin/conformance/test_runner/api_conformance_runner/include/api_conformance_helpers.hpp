// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "conformance.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace conformance {

inline const std::string get_plugin_lib_name_by_device(const std::string& deviceName) {
    const std::map<std::string, std::string> devices{
            { "AUTO", "openvino_auto_plugin" },
            { "HETERO", "openvino_hetero_plugin" },
            { "BATCH", "openvino_auto_batch_plugin" },
            { "MULTI", "openvino_auto_plugin" },
            { "HDDL", "openvino_intel_hddl_plugin" },
            { "VPUX", "openvino_intel_vpux_plugin" },
            { "CPU", "openvino_intel_cpu_plugin" },
            { "GNA", "openvino_intel_gna_plugin" },
            { "GPU", "openvino_intel_gpu_plugin" },
            { "MYRIAD", "openvino_intel_myriad_plugin" },
            { "TEMPLATE", "openvino_template_plugin" },
    };
    if (devices.find(deviceName) == devices.end()) {
        if (std::string(targetPluginName) != "") {
            return targetPluginName;
        }
        throw std::runtime_error("Incorrect device name");
    }
    return devices.at(deviceName);
}

inline const std::pair<std::string, std::string> generate_default_multi_config() {
    return {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::conformance::targetDevice};
}

inline const std::pair<std::string, std::string> generate_default_hetero_config() {
    return { "TARGET_FALLBACK" , ov::test::conformance::targetDevice };
}

inline const std::pair<std::string, std::string> generate_default_batch_config() {
    // auto-batching with batch 1 (no real batching in fact, but full machinery is in action)
    return { CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , ov::test::conformance::targetDevice };
}

inline const std::vector<std::map<std::string, std::string>> generate_configs(const std::string& target_plugin,
                                                                              const std::vector<std::map<std::string, std::string>>& config = {}) {
    std::pair<std::string, std::string> default_config;
    if (target_plugin ==  std::string(CommonTestUtils::DEVICE_MULTI) || target_plugin ==  std::string(CommonTestUtils::DEVICE_AUTO)) {
        default_config = generate_default_multi_config();
    } else if (target_plugin ==  std::string(CommonTestUtils::DEVICE_HETERO)) {
        default_config = generate_default_hetero_config();
    } else if (target_plugin ==  std::string(CommonTestUtils::DEVICE_BATCH)) {
        default_config = generate_default_batch_config();
    } else {
        throw std::runtime_error("Incorrect target device: " + target_plugin);
    }

    std::vector<std::map<std::string, std::string>> resultConfig;
    if (config.empty()) {
        return {{default_config}};
    }
    for (auto configItem : config) {
        configItem.insert(default_config);
        resultConfig.push_back(configItem);
    }
    return resultConfig;
}

inline const std::string generate_complex_device_name(const std::string& deviceName) {
    return deviceName + ":" + ov::test::conformance::targetDevice;
}

inline const std::vector<std::string> return_all_possible_device_combination() {
    std::vector<std::string> res{ov::test::conformance::targetDevice};
    std::vector<std::string> devices{CommonTestUtils::DEVICE_HETERO, CommonTestUtils::DEVICE_AUTO,
                                     CommonTestUtils::DEVICE_BATCH, CommonTestUtils::DEVICE_MULTI};
    for (const auto& device : devices) {
        res.emplace_back(generate_complex_device_name(device));
    }
    return res;
}

const std::vector<std::map<std::string, std::string>> empty_config = {
        {},
};

}  // namespace conformance
}  // namespace test
}  // namespace ov
