// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/range_add.hpp"

#include <vector>

using namespace ov::test;

namespace {

const std::vector<float> positiveStart = {1.0f, 1.2f};
const std::vector<float> positiveStop = {5.0f, 5.2f};
const std::vector<float> positiveStep = {1.0f, 0.1f};

const std::vector<float> negativeStart = {1.0f, 1.2f};
const std::vector<float> negativeStop = {-5.0f, -5.2f};
const std::vector<float> negativeStep = {-1.0f, -0.1f};

const std::vector<float> trunc_start = {1.2f, 1.9f};
const std::vector<float> trunc_stop = {11.4f, 11.8f};
const std::vector<float> trunc_step = {1.3f, 2.8f};

const std::vector<ov::element::Type> element_types = {
    ov::element::f32,
    ov::element::f16  // "[NOT_IMPLEMENTED] Input image format FP16 is not supported yet...
};

// ------------------------------ V0 ------------------------------

INSTANTIATE_TEST_SUITE_P(smoke_BasicPositive,
                         RangeAddSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(positiveStart),
                                            ::testing::ValuesIn(positiveStop),
                                            ::testing::ValuesIn(positiveStep),
                                            ::testing::ValuesIn(element_types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RangeAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicNegative,
                         RangeAddSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(negativeStart),
                                            ::testing::ValuesIn(negativeStop),
                                            ::testing::ValuesIn(negativeStep),
                                            ::testing::ValuesIn(element_types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RangeAddSubgraphTest::getTestCaseName);

// ------------------------------ V4 ------------------------------
INSTANTIATE_TEST_SUITE_P(smoke_BasicPositive,
                         RangeNumpyAddSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(positiveStart),
                                            ::testing::ValuesIn(positiveStop),
                                            ::testing::ValuesIn(positiveStep),
                                            ::testing::ValuesIn(element_types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RangeNumpyAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicNegative,
                         RangeNumpyAddSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(negativeStart),
                                            ::testing::ValuesIn(negativeStop),
                                            ::testing::ValuesIn(negativeStep),
                                            ::testing::ValuesIn(element_types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RangeNumpyAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicTruncateInputs,
                         RangeNumpyAddSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(trunc_start),
                                            ::testing::ValuesIn(trunc_stop),
                                            ::testing::ValuesIn(trunc_step),
                                            ::testing::ValuesIn(element_types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RangeNumpyAddSubgraphTest::getTestCaseName);
}  // namespace
