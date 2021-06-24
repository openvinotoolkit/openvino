// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
    static const std::vector<ngraph::element::Type> precisionsTemplate = {
            ngraph::element::f32,
    };

    static const std::vector<std::size_t> batchSizesTemplate = {
            1, 2
    };

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(precisionsTemplate),
                                    ::testing::ValuesIn(batchSizesTemplate),
                                    ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE)),
                            LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
