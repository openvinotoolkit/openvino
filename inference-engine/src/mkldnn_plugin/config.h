// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>

namespace MKLDNNPlugin {

struct Config {
    Config() {
        updateProperties();
    }

    enum LPTransformsMode {
        Off,
        On,
    };

    enum InferenceThreadsBinding {NONE, CORES, NUMA} useThreadBinding = CORES;
    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    std::string dumpToDot = "";
    std::string dumpQuantizedGraphToDot = "";
    std::string dumpQuantizedGraphToIr = "";
    int batchLimit = 0;
    int throughputStreams = 1;
    int threadsNum = 0;
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();
    std::map<std::string, std::string> _config;
};

}  // namespace MKLDNNPlugin

