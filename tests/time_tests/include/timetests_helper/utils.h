// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/openvino.hpp>

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
 * @brief Function that enables Latency performance hint for specified device (OV API 2)
 */
void setPerformanceConfig(ov::Core ie, const std::string &device) {
    auto supported_config_keys = ie.get_property(device, ov::supported_properties);

    if (std::find(supported_config_keys.begin(), supported_config_keys.end(), ov::hint::performance_mode) ==
        supported_config_keys.end()) {
        std::cerr << "Device " << device << " doesn't support " << ov::hint::performance_mode.name() << " property!\n"
                  << "Performance config was not set.";
    }
    else
        ie.set_property(device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}
}
