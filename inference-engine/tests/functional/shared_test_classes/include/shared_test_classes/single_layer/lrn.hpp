// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        double,                        // Alpha
        double,                        // Beta
        double,                        // Bias
        size_t,                        // Size
        std::vector<int64_t>,          // Reduction axes
        InferenceEngine::Precision,    // Network precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::SizeVector,   // Input shapes
        std::string                    // Device name
> lrnLayerTestParamsSet;

class LrnLayerTest
        : public testing::WithParamInterface<lrnLayerTestParamsSet>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<lrnLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
