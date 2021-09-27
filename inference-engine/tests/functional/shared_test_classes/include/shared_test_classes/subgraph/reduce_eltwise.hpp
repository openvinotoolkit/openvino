// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {

using ReduceEltwiseParamsTuple = typename std::tuple<
        std::vector<size_t>,              // Input shapes
        std::vector<int>,                 // Axis to reduce order
        CommonTestUtils::OpType,          // Scalar or vector type axis
        bool,                             // Keep dims
        InferenceEngine::Precision,       // Network precision
        std::string>;                     // Device name

class ReduceEltwiseTest:
        public testing::WithParamInterface<ReduceEltwiseParamsTuple>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    std::shared_ptr<ngraph::Function> fn;
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceEltwiseParamsTuple> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
