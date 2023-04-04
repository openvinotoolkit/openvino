// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gna/gna_config.hpp>

#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

class GnaLayerTestCheck {
    bool verRead = false;
    int verMajor;
    int verMinor;
    std::string lastMsg;

public:
    void SetUp(const std::string deviceName) {
        InferenceEngine::Core ieCore;
        auto metrics = ieCore.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();

        if (deviceName == CommonTestUtils::DEVICE_GNA) {
            if (std::find(metrics.begin(), metrics.end(), METRIC_KEY(GNA_LIBRARY_FULL_VERSION)) != metrics.end()) {
                auto gnaLibVerStr =
                    ieCore.GetMetric(deviceName, METRIC_KEY(GNA_LIBRARY_FULL_VERSION)).as<std::string>();
                verRead = sscanf(gnaLibVerStr.c_str(), "%d.%d", &verMajor, &verMinor) == 2;
            }
        }
    }

    std::string& getLastCmpResultMsg() {
        return lastMsg;
    }

    bool gnaLibVersionLessThan(std::string verToCmp) {
        int verToCmpMajor;
        int verToCmpMinor;

        if (!verRead) {
            IE_THROW() << "GnaLayerTestCheck requires initialization with SetUp()";
        }

        if (sscanf(verToCmp.c_str(), "%d.%d", &verToCmpMajor, &verToCmpMinor) != 2) {
            return false;
        }

        if (verMajor < verToCmpMajor || (verMajor == verToCmpMajor && verMinor < verToCmpMinor)) {
            lastMsg = "GNA library version is less than " + verToCmp;
            return true;
        }
        return false;
    }
};
