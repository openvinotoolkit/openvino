// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "common/gna_target.hpp"
#include "descriptions/gna_flags.hpp"
#include "gna2-inference-api.h"
#include "ie_parameter.hpp"
#include "ie_precision.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {

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
        embedded_export_path = r.embedded_export_path;
        dumpXNNGeneration = r.dumpXNNGeneration;
        target = std::make_shared<common::Target>();
        if (r.target) {
            *target = *r.target;
        }
        pluginGna2AccMode = r.pluginGna2AccMode;
        swExactMode = r.swExactMode;
        inputScaleFactorsPerInput = r.inputScaleFactorsPerInput;
        inputScaleFactors = r.inputScaleFactors;
        gnaFlags = r.gnaFlags;
        std::lock_guard<std::mutex> lock(r.mtx4keyConfigMap);
        keyConfigMap = r.keyConfigMap;
        cacheDir = r.cacheDir;
    }
    void UpdateFromMap(const std::map<std::string, std::string>& configMap);
    void AdjustKeyMapValues();
    InferenceEngine::Parameter GetParameter(const std::string& name) const;
    std::vector<std::string> GetSupportedKeys() const;
    static const InferenceEngine::Parameter GetImpactingModelCompilationProperties(bool compiled);
    static const InferenceEngine::Parameter GetSupportedProperties(bool compiled = false);

    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::UNDEFINED;

    // default precision of GNA hardware model
    ov::element::Type inference_precision = ov::element::undefined;
    InferenceEngine::Precision gnaPrecision = InferenceEngine::Precision::I16;

    std::string embedded_export_path;
    std::string dumpXNNGeneration;

    std::shared_ptr<common::Target> target = std::make_shared<common::Target>();

    Gna2AccelerationMode pluginGna2AccMode = Gna2AccelerationModeSoftware;
    bool swExactMode = true;

    std::map<std::string, float> inputScaleFactorsPerInput;
    std::vector<float> inputScaleFactors;  // Legacy one, should be removed with old confg API
    GNAFlags gnaFlags;

    mutable std::mutex mtx4keyConfigMap;
    std::map<std::string, std::string> keyConfigMap;
    static const uint8_t max_num_requests = 127;
    std::string cacheDir;
};

}  // namespace intel_gna
}  // namespace ov
