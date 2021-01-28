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
        int64_t,                       // depth
        float,                         // on_value
        float,                         // off_value
        int64_t,                       // axis
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> oneHotLayerTestParamsSet;

class OneHotLayerTest : public testing::WithParamInterface<oneHotLayerTestParamsSet>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<oneHotLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
