// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "config.h"
#include "ie_plugin_config.hpp"
#include "ie_common.h"

#include <string>
#include <map>
#include <algorithm>
#include <cpp_interfaces/exception2status.hpp>

namespace MKLDNNPlugin {

using namespace InferenceEngine;

void Config::readProperties(const std::map<std::string, std::string> &prop) {
    for (auto& kvp : prop) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        if (key == PluginConfigParams::KEY_CPU_BIND_THREAD) {
            if (val == PluginConfigParams::YES) useThreadBinding = true;
            else if (val == PluginConfigParams::NO) useThreadBinding = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_CPU_BIND_THREAD
                                   << ". Expected only YES/NO";
        } else if (key == PluginConfigParams::KEY_DYN_BATCH_LIMIT) {
            int val_i = std::stoi(val);
            // zero and any negative value will be treated
            // as default batch size
            batchLimit = std::max(val_i, 0);
        } else if (key == PluginConfigParams::KEY_PERF_COUNT) {
            if (val == PluginConfigParams::YES) collectPerfCounters = true;
            else if (val == PluginConfigParams::NO) collectPerfCounters = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_PERF_COUNT
                                   << ". Expected only YES/NO";
        } else if (key == PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) {
            if (val == PluginConfigParams::YES) exclusiveAsyncRequests = true;
            else if (val == PluginConfigParams::NO) exclusiveAsyncRequests = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS
                                   << ". Expected only YES/NO";
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0)
                enableDynamicBatch = true;
            else if (val.compare(PluginConfigParams::NO) == 0)
                enableDynamicBatch = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_DYN_BATCH_ENABLED
                << ". Expected only YES/NO";
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property " << key << " by CPU plugin";
        }
    }
}

}  // namespace MKLDNNPlugin
