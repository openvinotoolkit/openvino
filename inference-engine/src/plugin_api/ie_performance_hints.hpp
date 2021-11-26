// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for config that holds the performance hints
 * @file ie_performance_hints.hpp
 */

#pragma once
#include <ie_parameter.hpp>
#include <ie_plugin_config.hpp>

namespace InferenceEngine {
struct PerfHintsConfig {
    std::string ovPerfHint = "";
    int ovPerfHintNumRequests = 0;

    /**
     * @brief Parses configuration key/value pair
     * @param key configuration key
     * @param value configuration values
     */
    void SetConfig(const std::string& key, const std::string& value) {
        if (PluginConfigParams::KEY_PERFORMANCE_HINT == key) {
            ovPerfHint = StrictlyCheckPerformanceHintValue(value);
        } else if (PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS == key) {
            ovPerfHintNumRequests = StrictlyCheckPerformanceHintRequestValue(value);
        } else {
            IE_THROW() << "Unsupported Performance Hint config: " << key << std::endl;
        }
    }

    /**
     * @brief Return configuration value
     * @param key configuration key
     * @return configuration value wrapped into Parameter
     */
    Parameter GetConfig(const std::string& key) {
        if (PluginConfigParams::KEY_PERFORMANCE_HINT == key) {
            return ovPerfHint;
        } else if (PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS == key) {
            return ovPerfHintNumRequests;
        } else {
            IE_THROW() << "Unsupported Performance Hint config: " << key << std::endl;
        }
    }

    /**
     * @brief Supported Configuration keys
     * @return vector of supported configuration keys
     */
    static std::vector<std::string> SupportedKeys() {
        return {PluginConfigParams::KEY_PERFORMANCE_HINT, PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS};
    }

    /**
     * @brief Checks configuration key and value, otherwise throws
     * @param configuration key + value
     * @return void
     */
    static void CheckConfigAndValue(std::pair<const std::string, const std::string&> kvp) {
        if (kvp.first == PluginConfigParams::KEY_PERFORMANCE_HINT)
            StrictlyCheckPerformanceHintValue(kvp.second);
        else if (kvp.first == PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS)
            StrictlyCheckPerformanceHintRequestValue(kvp.second);
        else
            IE_THROW() << "Unsupported Performance Hint config: " << kvp.first << std::endl;
    }

    /**
     * @brief Returns configuration value if it is valid, otherwise throws
     * @param configuration value
     * @return configuration value
     */
    static std::string StrictlyCheckPerformanceHintValue(const std::string& val) {
        if (val == PluginConfigParams::LATENCY || val == PluginConfigParams::THROUGHPUT)
            return val;
        else
            IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_PERFORMANCE_HINT
                       << ". Expected only " << PluginConfigParams::LATENCY << "/" << PluginConfigParams::THROUGHPUT;
    }

    /**
     * @brief Returns configuration value if it is valid, otherwise throws
     * @param configuration value as string
     * @return configuration value as number
     */
    static int CheckPerformanceHintRequestValue(const std::string& val) {
        try {
            int val_i = std::stoul(val);
            if (val_i >= 0) {
                return val_i;
            }
        } catch (const std::logic_error&) {
        }
        IE_THROW() << "Wrong value of " << val << " for property key "
                   << PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS
                   << ". Expected only positive integer numbers";
    }

    /**
     * @brief Returns configuration value if it can be set, otherwise throws
     * @param configuration value as string
     * @return configuration value as number
     */
    static int StrictlyCheckPerformanceHintRequestValue(const std::string& val) {
        int val_i = CheckPerformanceHintRequestValue(val);
        if (val_i == 0) {
            IE_THROW() << "Wrong value of " << val << " for property key "
                       << PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS
                       << ". Expected only positive integer numbers";
        }
        return val_i;
    }
};
}  // namespace InferenceEngine
