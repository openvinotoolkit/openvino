// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "conformance.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace conformance {

inline const std::vector<ov::AnyMap> generate_ov_configs(const std::vector<ov::AnyMap>& config = {}) {
    std::vector<ov::AnyMap> resultConfig;
    for (auto configItem : config) {
        configItem.insert(ov::test::conformance::pluginConfig.begin(), ov::test::conformance::pluginConfig.end());
        resultConfig.push_back(configItem);
    }
    return resultConfig;
}

inline const std::vector<std::string> get_device(const std::string& targetDevice) {
    if (ov::test::conformance::targetDevice != targetDevice) {
        return {};
    }

    return { ov::test::conformance::targetDevice };
}

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

inline std::vector<std::pair<std::string, std::string>> generate_ov_pairs_plugin_name_by_device() {
    std::vector<std::pair<std::string, std::string>> res;
    std::string device(ov::test::conformance::targetDevice);
    std::string real_device = device.substr(0, device.find(':'));
    res.push_back(std::make_pair(get_plugin_lib_name_by_device(real_device),
                                    real_device));
    return res;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
