// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
        InferenceEngine::SizeVector,   // padsBegin
        InferenceEngine::SizeVector,   // padsEnd
        float,                         // argPadValue
        ngraph::helpers::PadMode,      // padMode
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> padLayerTestParamsSet;

namespace LayerTestsDefinitions {

class PadLayerTest : public testing::WithParamInterface<padLayerTestParamsSet>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<padLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions