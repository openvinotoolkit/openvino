// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/constant_result.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;
using namespace InferenceEngine;

namespace {

const std::vector<ConstantSubgraphType> types = {
    ConstantSubgraphType::SINGLE_COMPONENT,
    ConstantSubgraphType::SEVERAL_COMPONENT
};

const std::vector<SizeVector> shapes = {
    {1, 3, 10, 10},
    {2, 3, 4, 5}
};

const std::vector<Precision> precisions = {
    Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ConstantResultSubgraphTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(types),
                            ::testing::ValuesIn(shapes),
                            ::testing::ValuesIn(precisions),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ConstantResultSubgraphTest::getTestCaseName);

}  // namespace

