// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if GNA_LIB_VER == 1
#include <gna-api.h>
#else
#include <gna2-inference-api.h>
#include <gna2-common-api.h>
#endif
#include "ie_precision.hpp"
#include "descriptions/gna_flags.hpp"
#include <vector>
#include <map>
#include <mutex>

namespace GNAPluginNS {

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
        gnaPrecision = r.gnaPrecision;
        dumpXNNPath = r.dumpXNNPath;
        dumpXNNGeneration = r.dumpXNNGeneration;
#if GNA_LIB_VER == 1
        gna_proc_type = r.gna_proc_type;
#else
        pluginGna2AccMode = r.pluginGna2AccMode;
        swExactMode = r.swExactMode;
#endif
        inputScaleFactors = r.inputScaleFactors;
        gnaFlags = r.gnaFlags;
        std::lock_guard<std::mutex> lock(r.mtx4keyConfigMap);
        keyConfigMap = r.keyConfigMap;
    }
    void UpdateFromMap(const std::map<std::string, std::string>& configMap);
    void AdjustKeyMapValues();
    std::string GetParameter(const std::string& name) const;
    std::vector<std::string> GetSupportedKeys() const;

    // default precision of GNA hardware model (see QuantI16 quantizer struct)
    InferenceEngine::Precision gnaPrecision = InferenceEngine::Precision::I16;

    std::string dumpXNNPath;
    std::string dumpXNNGeneration;

    std::string gnaExecTarget;
    std::string gnaCompileTarget;

#if GNA_LIB_VER == 1
    intel_gna_proc_t gna_proc_type = static_cast<intel_gna_proc_t>(GNA_SOFTWARE & GNA_HARDWARE);
#else
    Gna2AccelerationMode pluginGna2AccMode = Gna2AccelerationModeSoftware;
    bool swExactMode = true;
#endif

    std::vector<float> inputScaleFactors;
    GNAFlags gnaFlags;

    mutable std::mutex mtx4keyConfigMap;
    std::map<std::string, std::string> keyConfigMap;
};

}  // namespace GNAPluginNS
