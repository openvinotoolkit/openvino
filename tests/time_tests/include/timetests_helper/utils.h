// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <ie_plugin_config.hpp>

#include <string>

namespace TimeTest {
/**
* @brief Get extension from filename
* @param filename - name of the file which extension should be extracted
* @return string with extracted file extension
*/
std::string fileExt(const std::string &filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

/**
 * @brief Function that enables Latency performance hint for specified device (OV API 1)
 */
void setPerformanceConfig(InferenceEngine::Core ie, const std::string &device) {
    auto supported_config_keys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();

    if (std::find(supported_config_keys.begin(), supported_config_keys.end(), "PERFORMANCE_HINT") ==
        supported_config_keys.end()) {
        std::cerr << "Device " << device << " doesn't support config key 'PERFORMANCE_HINT'!\n"
                  << "Performance config was not set.";
    }
    else
        ie.SetConfig({{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}}, device);
}

/**
 * @brief Function that enables Latency performance hint for specified device (OV API 2)
 */
void setPerformanceConfig(ov::Core ie, const std::string &device) {
    auto supported_config_keys = ie.get_property(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();

    if (std::find(supported_config_keys.begin(), supported_config_keys.end(), "PERFORMANCE_HINT") ==
        supported_config_keys.end()) {
        std::cerr << "Device " << device << " doesn't support config key 'PERFORMANCE_HINT'!\n"
                  << "Performance config was not set.";
    }
    else
        ie.set_property(device, {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}});
}
}
