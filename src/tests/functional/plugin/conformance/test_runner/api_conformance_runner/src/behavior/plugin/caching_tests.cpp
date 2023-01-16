// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/caching_tests.hpp"
#include <ov_ops/nms_ie_internal.hpp>
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace LayerTestsDefinitions;
using namespace ngraph;

static const std::vector<ov::element::Type> precisionsTemplate = {
        ov::element::f64,
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
        ov::element::i32,
        ov::element::i16,
        ov::element::i8,
        ov::element::u64,
        ov::element::u32,
        ov::element::u16,
        ov::element::u8,
        ov::element::boolean,
};

static const std::vector<std::size_t> batchSizesTemplate = {
        1, 2
};

static const std::vector<ov::element::Type> numericPrecisionsTemplate(precisionsTemplate.begin(),
                                                                      precisionsTemplate.end() - 1);

static const std::vector<ov::element::Type> floatingPointPrecisionsTemplate(precisionsTemplate.begin(),
                                                                            precisionsTemplate.begin() + 3);

INSTANTIATE_TEST_SUITE_P(ie_plugin_any_type, LoadNetworkCacheTestBase,
                         ::testing::Combine(
                                 ::testing::ValuesIn(LoadNetworkCacheTestBase::getAnyTypeOnlyFunctions()),
                                 ::testing::ValuesIn(precisionsTemplate),
                                 ::testing::ValuesIn(batchSizesTemplate),
                                 ::testing::ValuesIn(return_all_possible_device_combination())),
                         LoadNetworkCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ie_plugin_numeric, LoadNetworkCacheTestBase,
                         ::testing::Combine(
                                 ::testing::ValuesIn(LoadNetworkCacheTestBase::getNumericTypeOnlyFunctions()),
                                 ::testing::ValuesIn(numericPrecisionsTemplate),
                                 ::testing::ValuesIn(batchSizesTemplate),
                                 ::testing::ValuesIn(return_all_possible_device_combination())),
                         LoadNetworkCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ie_plugin_float, LoadNetworkCacheTestBase,
                         ::testing::Combine(
                                 ::testing::ValuesIn(LoadNetworkCacheTestBase::getFloatingPointOnlyFunctions()),
                                 ::testing::ValuesIn(floatingPointPrecisionsTemplate),
                                 ::testing::ValuesIn(batchSizesTemplate),
                                 ::testing::ValuesIn(return_all_possible_device_combination())),
                         LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
