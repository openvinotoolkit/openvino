// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-inference-api.h>
#include <gna2-common-api.h>
#include "openvino/runtime/intel_gna/properties.hpp"
#include "ie_precision.hpp"
#include <ie_parameter.hpp>
#include "descriptions/gna_flags.hpp"
#include <vector>
#include <map>
#include <mutex>

namespace GNAPluginNS {

static const float kScaleFactorDefault = 1.f;

struct Config {
    Config() {
        AdjustKeyMapValues();
    }
    Config(const Config& r) {
        Copy(r);
    }
    Config& operator=(const Config& r) {
        Copy(r);
        return *this;
    }
    void Copy(const Config& r) {
        performance_mode = r.performance_mode;
        inference_precision = r.inference_precision;
        gnaPrecision = r.gnaPrecision;
        dumpXNNPath = r.dumpXNNPath;
        dumpXNNGeneration = r.dumpXNNGeneration;
        gnaExecTarget = r.gnaExecTarget;
        gnaCompileTarget = r.gnaCompileTarget;
        pluginGna2AccMode = r.pluginGna2AccMode;
        swExactMode = r.swExactMode;
        inputScaleFactorsPerInput = r.inputScaleFactorsPerInput;
        inputScaleFactors = r.inputScaleFactors;
        gnaFlags = r.gnaFlags;
        std::lock_guard<std::mutex> lock(r.mtx4keyConfigMap);
        keyConfigMap = r.keyConfigMap;
    }
    void UpdateFromMap(const std::map<std::string, std::string>& configMap);
    void AdjustKeyMapValues();
    InferenceEngine::Parameter GetParameter(const std::string& name) const;
    std::vector<std::string> GetSupportedKeys() const;
    static const InferenceEngine::Parameter GetSupportedProperties(bool compiled = false);

    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::UNDEFINED;

    // default precision of GNA hardware model (see QuantI16 quantizer struct)
    ov::element::Type inference_precision = ov::element::undefined;
    InferenceEngine::Precision gnaPrecision = InferenceEngine::Precision::I16;

    std::string dumpXNNPath;
    std::string dumpXNNGeneration;

    std::string gnaExecTarget;
    std::string gnaCompileTarget;

    Gna2AccelerationMode pluginGna2AccMode = Gna2AccelerationModeSoftware;
    bool swExactMode = true;

    std::map<std::string, float> inputScaleFactorsPerInput;
    std::vector<float> inputScaleFactors; // Legacy one, should be removed with old confg API
    GNAFlags gnaFlags;

    mutable std::mutex mtx4keyConfigMap;
    std::map<std::string, std::string> keyConfigMap;

    static const uint8_t max_num_requests = 127;
};

}  // namespace GNAPluginNS
