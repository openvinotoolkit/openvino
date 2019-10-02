// Copyright (C) 2018-2019 Intel Corporation
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

    enum QuantizerMode {
        // Don't use Quantizer.
        None,
        // Handle quantize layers only on weights, execute in FP32.
        TransformQuantizeOnWeightsFp32,
        // Handle quantize layers only on weights, execute in INT8.
        TransformQuantizeOnWeightsInt8,
        // Handle quantize layers on weights and activations, execute in FP32.
        TransformQuantizeOnWeightsAndDataFp32,
        // Handle quantize layers on weights and activations, execute in INT8. Default value.
        TransformQuantizeOnWeightsAndDataInt8
    };

    bool useThreadBinding = true;
    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    std::string dumpToDot = "";
    std::string dumpQuantizedGraphToDot = "";
    std::string dumpQuantizedGraphToIr = "";
    int batchLimit = 0;
    int throughputStreams = 1;
    int threadsNum = 0;
    QuantizerMode quantizerMode = QuantizerMode::TransformQuantizeOnWeightsAndDataInt8;

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();
    std::map<std::string, std::string> _config;
};

}  // namespace MKLDNNPlugin

