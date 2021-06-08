// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.h"

#include <string>
#include <map>
#include <algorithm>

#include "ie_plugin_config.hpp"
#include "ie_common.h"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

namespace MKLDNNPlugin {

using namespace InferenceEngine;

Config::Config() {
    // this is default mode
    streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::CORES;

    // for the TBB code-path, additional configuration depending on the OS and CPU types
    #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        #if defined(__APPLE__) || defined(_WIN32)
        // 'CORES' is not implemented for Win/MacOS; so the 'NUMA' is default
        streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NUMA;
        #endif

        if (getAvailableCoresTypes().size() > 1 /*Hybrid CPU*/) {
            streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::HYBRID_AWARE;
        }
    #endif

    if (!with_cpu_x86_bfloat16())
        enforceBF16 = false;

    updateProperties();
}


void Config::readProperties(const std::map<std::string, std::string> &prop) {
    auto streamExecutorConfigKeys = streamExecutorConfig.SupportedKeys();
    for (auto& kvp : prop) {
        auto& key = kvp.first;
        auto& val = kvp.second;

        if (streamExecutorConfigKeys.end() !=
            std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streamExecutorConfig.SetConfig(key, val);
        } else if (key == PluginConfigParams::KEY_DYN_BATCH_LIMIT) {
            int val_i = -1;
            try {
                val_i = std::stoi(val);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_DYN_BATCH_LIMIT
                                    << ". Expected only integer numbers";
            }
            // zero and any negative value will be treated
            // as default batch size
            batchLimit = std::max(val_i, 0);
        } else if (key == PluginConfigParams::KEY_PERF_COUNT) {
            if (val == PluginConfigParams::YES) collectPerfCounters = true;
            else if (val == PluginConfigParams::NO) collectPerfCounters = false;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_PERF_COUNT
                                   << ". Expected only YES/NO";
        } else if (key == PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) {
            if (val == PluginConfigParams::YES) exclusiveAsyncRequests = true;
            else if (val == PluginConfigParams::NO) exclusiveAsyncRequests = false;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS
                                   << ". Expected only YES/NO";
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0)
                enableDynamicBatch = true;
            else if (val.compare(PluginConfigParams::NO) == 0)
                enableDynamicBatch = false;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_DYN_BATCH_ENABLED
                << ". Expected only YES/NO";
            IE_SUPPRESS_DEPRECATED_START
        } else if (key.compare(PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT) == 0) {
            IE_SUPPRESS_DEPRECATED_END
            // empty string means that dumping is switched off
            dumpToDot = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val == PluginConfigParams::NO)
                lpTransformsMode = LPTransformsMode::Off;
            else if (val == PluginConfigParams::YES)
                lpTransformsMode = LPTransformsMode::On;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE;
        } else if (key == PluginConfigParams::KEY_ENFORCE_BF16) {
            if (val == PluginConfigParams::YES) {
                if (with_cpu_x86_avx512_core()) {
                    enforceBF16 = true;
                    manualEnforceBF16 = true;
                } else {
                    IE_THROW() << "Platform doesn't support BF16 format";
                }
            } else if (val == PluginConfigParams::NO) {
                enforceBF16 = false;
                manualEnforceBF16 = false;
            } else {
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_ENFORCE_BF16
                    << ". Expected only YES/NO";
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property " << key << " by CPU plugin";
        }
        _config.clear();
    }
    if (exclusiveAsyncRequests)  // Exclusive request feature disables the streams
        streamExecutorConfig._streams = 1;

    updateProperties();
}
void Config::updateProperties() {
    if (!_config.size()) {
        switch (streamExecutorConfig._threadBindingType) {
            case IStreamsExecutor::ThreadBindingType::NONE:
                _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::NO });
            break;
            case IStreamsExecutor::ThreadBindingType::CORES:
                _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::YES });
            break;
            case IStreamsExecutor::ThreadBindingType::NUMA:
                _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::NUMA });
            break;
            case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
                _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::HYBRID_AWARE });
            break;
        }
        if (collectPerfCounters == true)
            _config.insert({ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::NO });
        if (exclusiveAsyncRequests == true)
            _config.insert({ PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::NO });
        if (enableDynamicBatch == true)
            _config.insert({ PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::NO });

        _config.insert({ PluginConfigParams::KEY_DYN_BATCH_LIMIT, std::to_string(batchLimit) });
        _config.insert({ PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, std::to_string(streamExecutorConfig._streams) });
        _config.insert({ PluginConfigParams::KEY_CPU_THREADS_NUM, std::to_string(streamExecutorConfig._threads) });
        IE_SUPPRESS_DEPRECATED_START
        _config.insert({ PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, dumpToDot });
        IE_SUPPRESS_DEPRECATED_END
        if (enforceBF16)
            _config.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO });
    }
}

}  // namespace MKLDNNPlugin
