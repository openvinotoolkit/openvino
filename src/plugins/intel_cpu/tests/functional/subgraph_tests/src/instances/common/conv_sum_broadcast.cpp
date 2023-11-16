// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/src/classes/conv_sum_broadcast.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
namespace ConvSumBroadcast {
INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_FP32, ConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape()),
                                 ::testing::ValuesIn(secondInp()),
                                 ::testing::Values(true, false),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_Several_Consumers, ConvSumInPlaceTestSeveralConsumers,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape()),
                                 ::testing::ValuesIn(secondInp()),
                                 ::testing::Values(true),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvSumInPlaceTest::getTestCaseName);

InputShape convInpShapeStrided = {
        //dynamic shapes
        {-1, 64, -1, -1},
        { //target static shapes
            {1, 64, 147, 147},
            {1, 64, 147, 147},
        }
};

InputShape secondInpStrided = {
        //dynamic shapes
        {-1, 128, -1, -1},
        { //target static shapes
            {1, 128, 74, 74},
            {1, 128, 74, 1}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_Strided, ConvSumInPlaceStrided,
                         ::testing::Combine(
                                 ::testing::Values(convInpShapeStrided),
                                 ::testing::Values(secondInpStrided),
                                 ::testing::Values(true),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvSumInPlaceTest::getTestCaseName);

} // namespace ConvSumBroadcast
} // namespace SubgraphTestsDefinitions
