// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/parameter_shapeof_result.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check, ParameterShapeOfResultSubgraphTest,
                        ::testing::Combine(
                            ::testing::Values(
                                ngraph::element::f32,
                                ngraph::element::f16),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ParameterShapeOfResultSubgraphTest::getTestCaseName);

}  // namespace
