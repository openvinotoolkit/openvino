// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"

using namespace ov::test::behavior;

namespace {
static const std::vector<ov::element::Type> precisionsTemplate = {
    ov::element::f32,
};

static const std::vector<std::size_t> batchSizesTemplate = {1, 2};

const std::vector<ov::AnyMap> autoConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::ValuesIn(precisionsTemplate),
                                            ::testing::ValuesIn(batchSizesTemplate),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         CompileModelCacheTestBase::getTestCaseName);

const std::vector<ov::AnyMap> LoadFromFileConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};
const std::vector<std::string> TestTargets = {
    ov::test::utils::DEVICE_AUTO,
    ov::test::utils::DEVICE_MULTI,
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase,
                         CompileModelLoadFromFileTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestTargets), ::testing::ValuesIn(LoadFromFileConfigs)),
                         CompileModelLoadFromFileTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase,
                         CompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestTargets), ::testing::ValuesIn(LoadFromFileConfigs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);
}  // namespace
