// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/src/classes/conv_maxpool_activ.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/filter_cpu_info.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {
namespace ConvPoolActiv {

const std::vector<fusingSpecificParams> fusingParamsSet_x64 {
        fusingSwish
};

INSTANTIATE_TEST_SUITE_P(smoke_Check_x64_Fusing, ConvPoolActivTest, ::testing::ValuesIn(fusingParamsSet_x64), ConvPoolActivTest::getTestCaseName);

} // namespace ConvPoolActiv
} // namespace SubgraphTestsDefinitions