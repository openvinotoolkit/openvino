// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gna/gna_config.hpp>

class GnaLayerTestCheck : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    bool skipTest = true;

    void SkipTestCheck() {
        InferenceEngine::Core ie_core;
        std::vector<std::string> metrics = ie_core.GetMetric(targetDevice, METRIC_KEY(SUPPORTED_METRICS));

        if (targetDevice == "GNA") {
            if (std::find(metrics.begin(), metrics.end(), METRIC_KEY(GNA_LIBRARY_FULL_VERSION)) != metrics.end()) {
                std::string gnaLibVer = ie_core.GetMetric(targetDevice, METRIC_KEY(GNA_LIBRARY_FULL_VERSION));

                if (gnaLibVer.rfind("2.1", 0) != 0) {
                    GTEST_SKIP() << "Disabled test due to GNA library version being < 2.1" << std::endl;
                }
                skipTest = false;
            }
        }
    }
};
