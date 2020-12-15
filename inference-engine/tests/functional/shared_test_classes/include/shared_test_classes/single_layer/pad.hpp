// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        std::vector<int64_t>,          // padsBegin
        std::vector<int64_t>,          // padsEnd
        float,                         // argPadValue
        ngraph::helpers::PadMode,      // padMode
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> padLayerTestParamsSet;

class PadLayerTest : public testing::WithParamInterface<padLayerTestParamsSet>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<padLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions