// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
    static const std::vector<ngraph::element::Type> precisionsMyriad = {
            ngraph::element::f32,
            ngraph::element::f16,
            ngraph::element::i32,
            ngraph::element::i8,
            ngraph::element::u8,
    };

    static const std::vector<std::size_t> batchSizesMyriad = {
            1, 2
    };

    INSTANTIATE_TEST_CASE_P(smoke_CachingSupportCase_Myriad, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(precisionsMyriad),
                                    ::testing::ValuesIn(batchSizesMyriad),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                            LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
