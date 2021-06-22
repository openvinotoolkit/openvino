// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <ie_precision.hpp>
#include <ie_common.h>
#include "../base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        size_t,                                                       // levels
        std::vector<size_t>,                                          // const inputs shape
        std::vector<float>                                            // input generator data: low, high, resolution
> fqSpecificParams;

typedef std::tuple<
        fqSpecificParams,
        InferenceEngine::SizeVector,                                  // Input shapes
        InferenceEngine::Precision,                                   // Input precision
        InferenceEngine::Precision,                                   // Axes precision
        std::vector<int>,                                             // Axes
        bool,                                                         // Normalize variance
        float,                                                        // Epsilon
        std::string,                                                  // Epsilon mode
        LayerTestsUtils::TargetDevice                                 // Device name
> fqSubgraphTestParamsSet;

class MvnFqMvnSubgraphTest : public testing::WithParamInterface<fqSubgraphTestParamsSet>,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqSubgraphTestParamsSet> obj);

protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    float inputDataMin        = 0.0;
    float inputDataMax        = 10.0;
    float inputDataResolution = 1.0;
    int32_t  seed = 1;
};
} // namespace SubgraphTestsDefinitions
