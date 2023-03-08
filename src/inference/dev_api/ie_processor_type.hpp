// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for config that holds the processor type for CPU inference
 * @file ie_processor_type.hpp
 */

#pragma once
#include <ie_parameter.hpp>
#include <ie_plugin_config.hpp>

namespace InferenceEngine {

struct ProcTypeConfig {
    std::string ovProcType = "CPU_DEFAULT";

    /**
     * @brief Parses configuration key/value pair
     * @param key configuration key
     * @param value configuration values
     */
    void SetConfig(const std::string& key, const std::string& value) {
        if ((CPUConfigParams::KEY_CPU_PROCESSOR_TYPE == key) &&
            (value == CPUConfigParams::CPU_DEFAULT || value == CPUConfigParams::CPU_ALL_CORE ||
             value == CPUConfigParams::CPU_PHY_CORE_ONLY || value == CPUConfigParams::CPU_P_CORE_ONLY ||
             value == CPUConfigParams::CPU_E_CORE_ONLY || value == CPUConfigParams::CPU_PHY_P_CORE_ONLY)) {
            ovProcType = value;
        } else {
            IE_THROW() << "Wrong value " << value << "for property key " << CPUConfigParams::KEY_CPU_PROCESSOR_TYPE
                       << ". Expected only " << CPUConfigParams::CPU_ALL_CORE << "/"
                       << CPUConfigParams::CPU_PHY_CORE_ONLY << "/" << CPUConfigParams::CPU_P_CORE_ONLY << "/"
                       << CPUConfigParams::CPU_E_CORE_ONLY << "/" << CPUConfigParams::CPU_PHY_P_CORE_ONLY << std::endl;
        }
    }

    /**
     * @brief Return configuration value
     * @param key configuration key
     * @return configuration value wrapped into Parameter
     */
    Parameter GetConfig(const std::string& key) {
        if (CPUConfigParams::KEY_CPU_PROCESSOR_TYPE == key) {
            return ovProcType;
        } else {
            IE_THROW() << "Unsupported Processor Type config: " << key << std::endl;
        }
    }
};
}  // namespace InferenceEngine
