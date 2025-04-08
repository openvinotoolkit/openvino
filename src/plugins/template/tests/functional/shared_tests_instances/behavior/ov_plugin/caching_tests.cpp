// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

using namespace ov::test::behavior;

namespace {
static const std::vector<ov::element::Type> precisionsTemplate = {
    ov::element::f32,
};

static const std::vector<std::size_t> batchSizesTemplate = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_CachingSupportCase_Template,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::ValuesIn(precisionsTemplate),
                                            ::testing::ValuesIn(batchSizesTemplate),
                                            ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

const std::vector<ov::AnyMap> TemplateConfigs = {
    {ov::num_streams(2)},
};
const std::vector<std::string> TestTemplateTargets = {
    ov::test::utils::DEVICE_TEMPLATE,
};
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         CompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestTemplateTargets),
                                            ::testing::ValuesIn(TemplateConfigs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         CompileModelLoadFromCacheTest,
                         ::testing::Combine(::testing::ValuesIn(TestTemplateTargets),
                                            ::testing::ValuesIn(TemplateConfigs)),
                         CompileModelLoadFromCacheTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         CompileModelWithCacheEncryptionTest,
                         testing::ValuesIn(TestTemplateTargets),
                         CompileModelWithCacheEncryptionTest::getTestCaseName);
}  // namespace
