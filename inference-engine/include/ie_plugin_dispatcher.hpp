// Copyright (C) 2018 Intel Corporation
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
    explicit PluginDispatcher(const std::vector<file_name_t> &pp) : pluginDirs(pp) {}

    /**
    * @brief Loads a plugin from plugin directories
    * @param name Plugin name
    * @return A pointer to the loaded plugin
    */
    virtual InferencePlugin getPluginByName(const file_name_t& name) const {
        std::stringstream err;
        for (auto &pluginPath : pluginDirs) {
            try {
                return InferencePlugin(InferenceEnginePluginPtr(make_plugin_name(pluginPath, name)));
            }
            catch (const std::exception &ex) {
                err << "cannot load plugin: " << fileNameToString(name) << " from " << fileNameToString(pluginPath) << ": " << ex.what() << ", skipping\n";
            }
        }
        THROW_IE_EXCEPTION << "Plugin " << fileNameToString(name) << " cannot be loaded: " << err.str() << "\n";
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
                return getPluginByName(stringToFileName(name));
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
    file_name_t make_plugin_name(const file_name_t &path, const file_name_t &input) const {
        file_name_t separator =
#if defined _WIN32 || defined __CYGWIN__
#   if defined UNICODE
            L"\\";
#   else
            "\\";
#   endif
#else
            "/";
#endif
        if (path.empty())
            separator = file_name_t();
#ifdef _WIN32
        return path + separator + input +
#   if defined UNICODE
            L".dll";
#   else
            ".dll";
#   endif
#elif __APPLE__
        return path + separator + "lib" + input + ".dylib";
#else
        return path + separator + "lib" + input + ".so";
#endif
    }


private:
    std::vector<file_name_t> pluginDirs;
};
}  // namespace InferenceEngine
