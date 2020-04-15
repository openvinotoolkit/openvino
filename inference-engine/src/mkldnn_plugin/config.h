// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <threading/ie_istreams_executor.hpp>

namespace MKLDNNPlugin {

struct Config {
    Config() {
#if (defined(__APPLE__) || defined(_WIN32))
        streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NUMA;
#else
        streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::CORES;
#endif
        updateProperties();
    }

    enum LPTransformsMode {
        Off,
        On,
    };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    std::string dumpToDot = "";
    std::string dumpQuantizedGraphToDot = "";
    std::string dumpQuantizedGraphToIr = "";
    int batchLimit = 0;
    bool enforceBF16 = false;
    InferenceEngine::IStreamsExecutor::Config streamExecutorConfig;

#if defined(__arm__) || defined(__aarch64__)
    // Currently INT8 mode is not optimized on ARM, fallback to FP32 mode.
    LPTransformsMode lpTransformsMode = LPTransformsMode::Off;
#else
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;
#endif

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();
    std::map<std::string, std::string> _config;
};

}  // namespace MKLDNNPlugin
