// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/subgraph/preprocess.hpp"

using namespace SubgraphTestsDefinitions;

INSTANTIATE_TEST_SUITE_P(smoke_PrePostProcess, PrePostProcessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::builder::preprocess::generic_preprocess_functions()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         PrePostProcessTest::getTestCaseName);
