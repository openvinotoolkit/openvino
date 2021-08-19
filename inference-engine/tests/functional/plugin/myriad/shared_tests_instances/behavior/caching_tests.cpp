// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
    static const std::vector<ngraph::element::Type> nightly_precisionsMyriad = {
            ngraph::element::f32,
            ngraph::element::f16,
            ngraph::element::i32,
            ngraph::element::i8,
            ngraph::element::u8,
    };

    static const std::vector<ngraph::element::Type> smoke_precisionsMyriad = {
            ngraph::element::f32,
    };

    static const std::vector<std::size_t> batchSizesMyriad = {
            1, 2
    };

    static std::vector<nGraphFunctionWithName> smoke_functions() {
        auto funcs = LoadNetworkCacheTestBase::getStandardFunctions();
        if (funcs.size() > 1) {
            funcs.erase(funcs.begin() + 1, funcs.end());
        }
        return funcs;
    }

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Myriad, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(smoke_functions()),
                                    ::testing::ValuesIn(smoke_precisionsMyriad),
                                    ::testing::ValuesIn(batchSizesMyriad),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                            LoadNetworkCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(nightly_CachingSupportCase_Myriad, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(nightly_precisionsMyriad),
                                    ::testing::ValuesIn(batchSizesMyriad),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                            LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
