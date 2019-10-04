// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief A header for a class to handle plugin loading.
* @file ie_plugin_dispatcher.hpp
*/
#pragma once

#include "ie_plugin_ptr.hpp"
#include <string>
#include <vector>
#include <cpp/ie_plugin_cpp.hpp>
#include <multi-device/multi_device_config.hpp>

namespace InferenceEngine {
/**
* @brief This is a class to load a suitable plugin
*/
class INFERENCE_ENGINE_API_CLASS(PluginDispatcher) {
public:
    /**
     * @brief A constructor
     * @param pp Vector of paths to plugin directories
     */
    explicit PluginDispatcher(const std::vector<file_name_t> &pp = {file_name_t()});

    /**
     * @brief Loads a plugin from plugin directories
     * @param name Plugin name
     * @return A pointer to the loaded plugin
     */
    virtual InferencePlugin getPluginByName(const file_name_t& name) const;

    /**
     * @deprecated Use InferenceEngine::Core to work with devices by name
     * @brief Loads a plugin from directories that is suitable for the device string
     * @param deviceName A string value representing target device
     * @return A pointer to the plugin
     */
    INFERENCE_ENGINE_DEPRECATED
    InferencePlugin getPluginByDevice(const std::string& deviceName) const;

    /**
     * @deprecated Use InferenceEngine::Core to work with devices by name
     * @brief Loads a plugin from directories that is suitable for the device
     * @param device An instance of InferenceEngine::TargetDevice
     * @return A pointer to the plugin
     */
    INFERENCE_ENGINE_DEPRECATED
    InferenceEnginePluginPtr getSuitablePlugin(TargetDevice device) const;

protected:
    /**
     * @brief Creates path to the plugin
     * @param path Path to the plugin
     * @param input Plugin name
     * @return The path to the plugin
     */
    file_name_t make_plugin_name(const file_name_t &path, const file_name_t &input) const;

private:
    std::vector<file_name_t> pluginDirs;
};
}  // namespace InferenceEngine
