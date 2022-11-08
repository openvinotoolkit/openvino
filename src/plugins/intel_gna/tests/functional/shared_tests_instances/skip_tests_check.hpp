// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gna/gna_config.hpp>

class GnaLayerTestCheck {
    float gnaLibVer = 0.0f;
    std::string lastMsg;

public:
    void SetUp(const std::string deviceName) {
        InferenceEngine::Core ieCore;
        std::vector<std::string> metrics = ieCore.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));

        if (deviceName == CommonTestUtils::DEVICE_GNA) {
            if (std::find(metrics.begin(), metrics.end(), METRIC_KEY(GNA_LIBRARY_FULL_VERSION)) != metrics.end()) {
                auto gnaLibVerStr =
                    ieCore.GetMetric(deviceName, METRIC_KEY(GNA_LIBRARY_FULL_VERSION)).as<std::string>();
                gnaLibVer = std::stof(gnaLibVerStr);
            }
        }
    }

    std::string& getLastCmpResultMsg() {
        return lastMsg;
    }

    bool gnaLibVersionLessThan(float verToCmp) {
        if (gnaLibVer && gnaLibVer < verToCmp) {
            lastMsg = "GNA library version is less than " + std::to_string(verToCmp);
            return true;
        }
        return false;
    }
};