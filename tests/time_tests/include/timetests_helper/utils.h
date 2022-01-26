// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
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
 * @brief Function that enables Latency performance hint for specified device.
 */
void setPerformanceConfig(InferenceEngine::Core ie, const std::string &device) {
  std::vector<std::string> supported_config_keys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));

  if (std::find(supported_config_keys.begin(), supported_config_keys.end(), "PERFORMANCE_HINT") ==
      supported_config_keys.end()) {
    std::cerr << "Device " << device << " doesn't support config key 'PERFORMANCE_HINT'!\n"
              << "Performance config was not set.";
  }
  else
    ie.SetConfig({{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}}, device);
}
}