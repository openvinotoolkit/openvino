// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        size_t,                         // Num splits
        size_t,                         // Axis
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        std::vector<size_t>,            // Input shapes
        std::string                     // Target device name
> splitParams;

class SplitLayerTest : public testing::WithParamInterface<splitParams>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions