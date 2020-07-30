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
        std::vector<int>,               // Axis to reduce order
        bool,                           // Keep dims
        ngraph::helpers::ReductionType, // Reduce operation type
        InferenceEngine::Precision,     // Net precision
        std::vector<size_t>,            // Input shapes
        std::string                     // Target device name
> reduceMeanParams;

class ReduceOpsLayerTest : public testing::WithParamInterface<reduceMeanParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reduceMeanParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions