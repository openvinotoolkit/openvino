// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for config that holds the performance hints
 * @file ie_performance_hints.hpp
 */

#pragma once
#include "ie_plugin_config.hpp"

namespace InferenceEngine {
    struct PerfHintsConfig {
        std::string ovPerfHint = "";
        int ovPerfHintNumRequests = 0;

        /**
        * @brief Supported Configuration keys
        * @return vector of supported configuration keys
        */
        std::vector<std::string> SupportedKeys() {
            return {PluginConfigParams::KEY_PERFORMANCE_HINT, PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS};
        }

        /**
        * @brief Parses configuration key/value pair
        * @param key configuration key
        * @param value configuration values
        */
        void SetConfig(const std::string& key, const std::string& value) {
            if (PluginConfigParams::KEY_PERFORMANCE_HINT == key) {
                ovPerfHint = CheckPerformanceHintValue(value);
            } else if (PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS == key) {
                ovPerfHint = CheckPerformanceHintRequestValue(value);
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
            }
        }

        /**
        * @brief Returns configuration value if it is valid, otherwise throws
        * @param configuration value
        * @return configuration value
        */
        static std::string CheckPerformanceHintValue(const std::string& val){
            if (val == PluginConfigParams::LATENCY || val == PluginConfigParams::THROUGHPUT)
                return val;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_PERFORMANCE_HINT
                << ". Expected only " << PluginConfigParams::LATENCY << "/" << PluginConfigParams::THROUGHPUT;
        }

        /**
        * @brief Returns configuration value if it is valid, otherwise throws
        * @param configuration value
        * @return configuration value
        */
        static int CheckPerformanceHintRequestValue(const std::string& val) {
            int val_i = -1;
            try {
                val_i = std::stoi(val);
                if (val_i > 0)
                    return val_i;
                else
                    throw std::logic_error("wrong val");
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value of " << val << " for property key "
                           << PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS << ". Expected only positive integer numbers";
            }
        }
    };
} // namespace InferenceEngine