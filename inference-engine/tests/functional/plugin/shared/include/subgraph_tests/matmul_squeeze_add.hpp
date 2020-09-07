// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

typedef std::tuple<
    InferenceEngine::Precision,          // Network Precision
    std::string,                         // Target Device
    std::map<std::string, std::string>   // Configuration
> matmulSqueezeAddParams;

namespace LayerTestsDefinitions {

class MatmulSqueezeAddTest : public testing::WithParamInterface<matmulSqueezeAddParams>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<matmulSqueezeAddParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
