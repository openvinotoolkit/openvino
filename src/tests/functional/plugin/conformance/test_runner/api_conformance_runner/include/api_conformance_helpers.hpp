// Copyright (C) 2018-2023 Intel Corporation
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
            { "NPU", "openvino_intel_npu_plugin" },
            { "CPU", "openvino_intel_cpu_plugin" },
            { "GNA", "openvino_intel_gna_plugin" },
            { "GPU", "openvino_intel_gpu_plugin" },
            { "TEMPLATE", "openvino_template_plugin" },
            { "NVIDIA", "openvino_nvidia_gpu_plugin" },
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
    if (target_plugin ==  std::string(ov::test::utils::DEVICE_MULTI) || target_plugin ==  std::string(ov::test::utils::DEVICE_AUTO)) {
        default_config = generate_default_multi_config();
    } else if (target_plugin ==  std::string(ov::test::utils::DEVICE_HETERO)) {
        default_config = generate_default_hetero_config();
    } else if (target_plugin ==  std::string(ov::test::utils::DEVICE_BATCH)) {
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
    if (deviceName == "BATCH") {
        return deviceName + ":" + ov::test::conformance::targetDevice + "(4)";
    }
    return deviceName + ":" + ov::test::conformance::targetDevice;
}

inline const std::vector<std::string> return_all_possible_device_combination(bool enable_complex_name = true) {
    std::vector<std::string> res{ov::test::conformance::targetDevice};
    std::vector<std::string> devices{ov::test::utils::DEVICE_HETERO, ov::test::utils::DEVICE_AUTO,
                                     ov::test::utils::DEVICE_BATCH, ov::test::utils::DEVICE_MULTI};
    for (const auto& device : devices) {
        res.emplace_back(enable_complex_name ? generate_complex_device_name(device) : device);
    }
    return res;
}

inline std::vector<std::pair<std::string, std::string>> generate_pairs_plugin_name_by_device() {
    std::vector<std::pair<std::string, std::string>> res;
    for (const auto& device : return_all_possible_device_combination()) {
        std::string real_device = device.substr(0, device.find(':'));
        res.push_back(std::make_pair(get_plugin_lib_name_by_device(real_device),
                                     real_device));
    }
    return res;
}

inline std::map<std::string, std::string> AnyMap2StringMap(const AnyMap& config) {
    if (config.empty())
        return {};
    std::map<std::string, std::string> result;
    for (const auto& configItem : config) {
        result.insert({configItem.first, configItem.second.as<std::string>()});
    }
    return result;
}

const std::map<std::string, std::string> ie_config = AnyMap2StringMap(ov::test::conformance::pluginConfig);

}  // namespace conformance
}  // namespace test
}  // namespace ov
