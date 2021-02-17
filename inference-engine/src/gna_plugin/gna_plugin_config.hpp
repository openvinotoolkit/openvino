// Copyright (C) 2020 Intel Corporation
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
        gnaPrecision = r.gnaPrecision;
        dumpXNNPath = r.dumpXNNPath;
        dumpXNNGeneration = r.dumpXNNGeneration;
#if GNA_LIB_VER == 1
        gna_proc_type = r.gna_proc_type;
#else
        pluginGna2AccMode = r.pluginGna2AccMode;
        pluginGna2DeviceConsistent = r.pluginGna2DeviceConsistent;
#endif
        inputScaleFactors = r.inputScaleFactors;
        gnaFlags = r.gnaFlags;
        std::lock_guard<std::mutex>(r.mtx4keyConfigMap);
        keyConfigMap = r.keyConfigMap;
    }
    void UpdateFromMap(const std::map<std::string, std::string>& configMap);
    void AdjustKeyMapValues();
    std::string GetParameter(const std::string& name) const;
    std::vector<std::string> GetSupportedKeys() const;

    // precision of GNA hardware model
    InferenceEngine::Precision gnaPrecision = InferenceEngine::Precision::I16;

    std::string dumpXNNPath;
    std::string dumpXNNGeneration;

#if GNA_LIB_VER == 1
    intel_gna_proc_t gna_proc_type = static_cast<intel_gna_proc_t>(GNA_SOFTWARE & GNA_HARDWARE);
#else
    Gna2AccelerationMode pluginGna2AccMode = Gna2AccelerationModeSoftware;
    Gna2DeviceVersion pluginGna2DeviceConsistent = Gna2DeviceVersion1_0;
#endif

    std::vector<float> inputScaleFactors;
    GNAFlags gnaFlags;

    mutable std::mutex mtx4keyConfigMap;
    std::map<std::string, std::string> keyConfigMap;
};

}  // namespace GNAPluginNS
