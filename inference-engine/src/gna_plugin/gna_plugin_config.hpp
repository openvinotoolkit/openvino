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

namespace GNAPluginNS {

struct Config {
    Config() {
        AdjustKeyMapValues();
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

    std::map<std::string, std::string> key_config_map;
};

}  // namespace GNAPluginNS
