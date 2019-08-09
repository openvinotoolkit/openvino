// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_dispatcher.hpp"
#include <string>
#include <vector>

using namespace InferenceEngine;

PluginDispatcher::PluginDispatcher(const std::vector<file_name_t> &pp) : pluginDirs(pp) {}

InferencePlugin PluginDispatcher::getPluginByName(const file_name_t& name) const {
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


IE_SUPPRESS_DEPRECATED_START

InferencePlugin PluginDispatcher::getPluginByDevice(const std::string& deviceName) const {
    InferenceEnginePluginPtr ptr;
    // looking for HETERO: if can find, add everything after ':' to the options of hetero plugin
    if (deviceName.find("HETERO:") == 0) {
        ptr = getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr("HETERO"));
        if (ptr) {
            InferenceEngine::ResponseDesc response;
            ptr->SetConfig({{"TARGET_FALLBACK", deviceName.substr(7, deviceName.length() - 7)}}, &response);
        }
    } else {
        ptr = getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr(deviceName));
    }
    return InferencePlugin(ptr);
}

InferenceEnginePluginPtr PluginDispatcher::getSuitablePlugin(TargetDevice device) const {
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

IE_SUPPRESS_DEPRECATED_END

file_name_t make_plugin_name(const file_name_t &path, const file_name_t &input) {
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

file_name_t PluginDispatcher::make_plugin_name(const file_name_t &path, const file_name_t &input) const {
    return ::make_plugin_name(path, input);
}
