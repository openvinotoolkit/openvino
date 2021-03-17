// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {

    INSTANTIATE_TEST_CASE_P(smoke_CachingSupportCase_Myriad, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::precisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                            LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
