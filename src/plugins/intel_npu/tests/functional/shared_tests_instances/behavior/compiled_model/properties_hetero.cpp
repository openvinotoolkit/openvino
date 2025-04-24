// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties_hetero.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, OVClassHeteroCompiledModelGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(ov::test::utils::DEVICE_NPU),
        ov::test::utils::appendPlatformTypeTestName<OVClassHeteroCompiledModelGetMetricTest_SUPPORTED_CONFIG_KEYS>);

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(ov::test::utils::DEVICE_NPU),
        ov::test::utils::appendPlatformTypeTestName<OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK>);

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES,
        ::testing::Values(ov::test::utils::DEVICE_NPU),
        ov::test::utils::appendPlatformTypeTestName<OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES>);

}  // namespace
