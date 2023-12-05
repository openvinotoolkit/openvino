// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
static const std::vector<ngraph::element::Type> precisionsGNA = {
    ngraph::element::f32,
    // integer weights are not supported by GNA so far
    // ngraph::element::u8,
    // ngraph::element::i16,
};

static const std::vector<std::size_t> batchSizesGNA = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GNA,
                         LoadNetworkCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                            ::testing::ValuesIn(precisionsGNA),
                                            ::testing::ValuesIn(batchSizesGNA),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         LoadNetworkCacheTestBase::getTestCaseName);
}  // namespace
