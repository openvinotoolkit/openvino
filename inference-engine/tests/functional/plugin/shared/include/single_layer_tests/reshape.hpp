// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using ShapeConfig = std::pair<
    InferenceEngine::SizeVector,  // input shape
    std::vector<int>              // resulting shape
>;

using reshapeParams = std::tuple<
    InferenceEngine::Precision,         // Input precision
    ShapeConfig,                        // Shapes config
    bool,                               // Special zero value
    bool,                               // Dyn Batch
    std::string                         // Device name
>;

class ReshapeLayerTest : public testing::WithParamInterface<reshapeParams>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reshapeParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions