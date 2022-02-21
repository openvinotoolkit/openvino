// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

using namespace ov::test::behavior;

namespace {
    static const std::vector<ngraph::element::Type> precisionsGNA = {
            ngraph::element::f32,
            // integer weights are not supported by GNA so far
            // ngraph::element::u8,
            // ngraph::element::i16,
    };

    static const std::vector<std::size_t> batchSizesGNA = {
            1, 2
    };

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GNA, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(precisionsGNA),
                                    ::testing::ValuesIn(batchSizesGNA),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                            CompileModelCacheTestBase::getTestCaseName);
} // namespace
