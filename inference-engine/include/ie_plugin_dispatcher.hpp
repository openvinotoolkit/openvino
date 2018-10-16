// Copyright (C) 2018 Intel Corporation
//
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

namespace InferenceEngine {
/**
* @brief This is a class to load a suitable plugin
*/
class PluginDispatcher {
public:
    /**
     * @brief A constructor
     * @param pp Vector of paths to plugin directories
     */
    explicit PluginDispatcher(const std::vector<std::string> &pp) : pluginDirs(pp) {}

    /**
    * @brief Loads a plugin from plugin directories
    * @param name Plugin name
    * @return A pointer to the loaded plugin
    */
    virtual InferencePlugin getPluginByName(const std::string& name) const {
        std::stringstream err;
        for (auto &pluginPath : pluginDirs) {
            try {
                return InferencePlugin(InferenceEnginePluginPtr(make_plugin_name(pluginPath, name)));
            }
            catch (const std::exception &ex) {
                err << "cannot load plugin: " << name << " from " << pluginPath << ": " << ex.what() << ", skipping\n";
            }
        }
        THROW_IE_EXCEPTION << "Plugin " << name << " cannot be loaded: " << err.str() << "\n";
    }

    /**
    * @brief Loads a plugin from directories that is suitable for the device string
    * @return A pointer to the plugin
    */
    InferencePlugin getPluginByDevice(const std::string& deviceName) const {
        InferenceEnginePluginPtr ptr;
        // looking for HETERO: if can find, add everything after ':' to the options of hetero plugin
        if (deviceName.find("HETERO:") == 0) {
            ptr = getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr("HETERO"));
            if (ptr) {
                InferenceEngine::ResponseDesc response;
                ptr->SetConfig({ { "TARGET_FALLBACK", deviceName.substr(7, deviceName.length() - 7) } }, &response);
            }
        } else {
            ptr = getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr(deviceName));
        }
        return InferencePlugin(ptr);
    }

    /**
    * @brief Loads a plugin from directories that is suitable for the device
    * @return A pointer to the plugin
    */
    InferenceEnginePluginPtr getSuitablePlugin(TargetDevice device) const {
        FindPluginResponse result;
        ResponseDesc desc;
        if (InferenceEngine::OK != findPlugin({ device }, result, &desc)) {
            THROW_IE_EXCEPTION << desc.msg;
        }

        std::stringstream err;
        for (std::string& name : result.names) {
            try {
                return getPluginByName(name);
            }
            catch (const std::exception &ex) {
                err << "Tried load plugin : " << name << ",  error: " << ex.what() << "\n";
            }
        }
        THROW_IE_EXCEPTION << "Cannot find plugin to use :" << err.str() << "\n";
    }

protected:
    /**
    * @brief Creates path to the plugin
    * @param path Path to the plugin
    * @param input Plugin name
    * @return The path to the plugin
    */
    std::string make_plugin_name(const std::string &path, const std::string &input) const {
        std::string separator =
#if defined _WIN32 || defined __CYGWIN__
        "\\";
#else
        "/";
#endif
        if (path.empty())
            separator = "";
#ifdef _WIN32
        return path + separator + input + ".dll";
#elif __APPLE__
        return path + separator + "lib" + input + ".dylib";
#else
        return path + separator + "lib" + input + ".so";
#endif
    }

private:
    std::vector<std::string> pluginDirs;
};
}  // namespace InferenceEngine
