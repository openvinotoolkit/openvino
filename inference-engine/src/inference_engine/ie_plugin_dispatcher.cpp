// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_dispatcher.hpp"

#include <map>
#include <multi-device/multi_device_config.hpp>
#include <string>
#include <vector>

#include "file_utils.h"

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

PluginDispatcher::PluginDispatcher(const std::vector<file_name_t>& pp): pluginDirs(pp) {}

InferencePlugin PluginDispatcher::getPluginByName(const file_name_t& name) const {
    std::stringstream err;
    for (auto& pluginPath : pluginDirs) {
        try {
            return InferencePlugin(InferenceEnginePluginPtr(FileUtils::makeSharedLibraryName(pluginPath, name)));
        } catch (const std::exception& ex) {
            err << "cannot load plugin: " << fileNameToString(name) << " from " << fileNameToString(pluginPath) << ": "
                << ex.what() << ", skipping\n";
        }
    }
    THROW_IE_EXCEPTION << "Plugin " << fileNameToString(name) << " cannot be loaded: " << err.str() << "\n";
}

namespace {

std::string getPluginName(const std::string& deviceName) {
    static std::map<std::string, std::string> plugunFromDeviceMap = {
        {"CPU", "MKLDNNPlugin"},    {"GPU", "clDNNPlugin"},         {"FPGA", "dliaPlugin"},
        {"MYRIAD", "myriadPlugin"}, {"HDDL", "HDDLPlugin"},         {"GNA", "GNAPlugin"},
        {"HETERO", "HeteroPlugin"}, {"MULTI", "MultiDevicePlugin"}, {"KMB", "kmbPlugin"}};
    auto val = plugunFromDeviceMap.find(deviceName);

    if (val == plugunFromDeviceMap.end()) {
        THROW_IE_EXCEPTION << "Cannot find plugin name for device " << deviceName;
    }

    return val->second;
}

}  // namespace

InferencePlugin PluginDispatcher::getPluginByDevice(const std::string& deviceName) const {
    auto createPluginByDevice = [&](const std::string& deviceName) {
        std::string pluginName = getPluginName(deviceName);

        std::stringstream err;
        try {
            return getPluginByName(stringToFileName(pluginName));
        } catch (const std::exception& ex) {
            err << "Tried load plugin : " << pluginName << " for device " << deviceName << ",  error: " << ex.what()
                << "\n";
        }

        THROW_IE_EXCEPTION << "Cannot find plugin to use: " << err.str() << "\n";
    };

    InferenceEnginePluginPtr ptr;
    // looking for HETERO: if can find, add everything after ':' to the options of hetero plugin
    if (deviceName.find("HETERO:") == 0) {
        ptr = createPluginByDevice("HETERO");
        if (ptr) {
            InferenceEngine::ResponseDesc response;
            ptr->SetConfig({{"TARGET_FALLBACK", deviceName.substr(7, deviceName.length() - 7)}}, &response);
        }
    } else if (deviceName.find("MULTI:") == 0) {
        // MULTI found: everything after ':' to the options of the multi-device plugin
        ptr = createPluginByDevice("MULTI");
        if (ptr) {
            InferenceEngine::ResponseDesc response;
            if (deviceName.length() < 6) THROW_IE_EXCEPTION << "Missing devices priorities for the multi-device case";
            ptr->SetConfig({{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
                             deviceName.substr(6, deviceName.length() - 6)}},
                           &response);
        }
    } else {
        ptr = createPluginByDevice(deviceName);
    }
    return InferencePlugin(ptr);
}

file_name_t PluginDispatcher::make_plugin_name(const file_name_t& path, const file_name_t& input) const {
    return FileUtils::makeSharedLibraryName(path, input);
}

IE_SUPPRESS_DEPRECATED_END
