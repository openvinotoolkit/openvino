// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Provides parser for device name
 * @file openvino/runtime/device_id_parser.hpp
 */

#pragma once

#include <string>

#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @brief Class parses device name and id
 */
class OPENVINO_RUNTIME_API DeviceIDParser {
    std::string m_device_name;
    std::string m_device_id;

public:
    explicit DeviceIDParser(const std::string& device_name_with_id);

    const std::string& get_device_id() const;
    const std::string& get_device_name() const;

    static std::vector<std::string> get_hetero_devices(const std::string& fallbackDevice);
    static std::vector<std::string> get_multi_devices(const std::string& devicesList);
    static std::string get_batch_device(const std::string& devicesList);
};

}  // namespace ov
