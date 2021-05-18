// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
    std::pair<InferenceEngine::SizeVector, InferenceEngine::SizeVector>,  // Input shape, Constant shape
    InferenceEngine::Precision,                                           // Data precision
    InferenceEngine::Precision,                                           // Axes precision
    std::vector<int>,                                                     // Axes
    bool,                                                                 // Normalize variance
    float,                                                                // Epsilon
    std::string,                                                          // Epsilon mode
    std::string                                                           // Device name
> mvnMultiplyAddParams;

class MVNMultiplyAdd: public testing::WithParamInterface<mvnMultiplyAddParams>,
                      public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvnMultiplyAddParams> &obj);
protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
