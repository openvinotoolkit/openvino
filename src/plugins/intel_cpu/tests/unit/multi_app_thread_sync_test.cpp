// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "config.h"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

using namespace ov::intel_cpu;

namespace {

TEST(MultiAppThreadSyncTest, DefaultMultiAppThreadSyncIsFalse) {
    Config config;
    ASSERT_FALSE(config.multiAppThreadSyncExecution);
}

TEST(MultiAppThreadSyncTest, ReadPropertiesSetTrue) {
    Config config;
    OV_ASSERT_NO_THROW(config.readProperties(
        {{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}}));
    ASSERT_TRUE(config.multiAppThreadSyncExecution);
}

TEST(MultiAppThreadSyncTest, ReadPropertiesSetFalse) {
    Config config;
    config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}});
    ASSERT_TRUE(config.multiAppThreadSyncExecution);

    OV_ASSERT_NO_THROW(config.readProperties(
        {{ov::intel_cpu::multi_app_thread_sync_execution.name(), false}}));
    ASSERT_FALSE(config.multiAppThreadSyncExecution);
}

TEST(MultiAppThreadSyncTest, ReadPropertiesInvalidValueThrows) {
    Config config;
    ASSERT_THROW(
        config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(),
                                std::string("maybe")}}),
        ov::Exception);
}

// NOTE: `EffectiveValueFollowsRequestedValue` / `EffectiveValueIsFalseWhenNotRequested` used to
// live here, directly exercising `Config::normalizeMultiAppThreadSyncExecution()` and
// `config.runSyncInferInCallerThread`. Per the review-fix plan for PR #35276, both that method
// and that member were removed: the "is caller-thread-sync actually safe/effective right now"
// computation was moved out of `Config` (which only ever stores the raw user-requested
// `multiAppThreadSyncExecution` value) and into `CompiledModel`'s constructor, where it can also
// see `m_has_sub_compiled_models` / `exclusiveAsyncRequests` and the real task executor type --
// none of which `Config` alone has visibility into. That computation is already covered
// end-to-end through the public API by the functional tests in
// tests/functional/custom/behavior/ov_executable_network/properties.cpp
// (smoke_CpuExecNetworkCheckMultiAppThreadSyncExecutionWithZeroStreams,
// smoke_CpuExecNetworkCheckExclusiveAsyncDisablesMultiAppThreadSyncExecution,
// smoke_CpuExecNetworkCheckTensorParallelRejectsMultiAppThreadSyncExecution,
// smoke_CpuExecNetworkMultiAppThreadSync{DefaultIsFalse,SetTrue,SetFalse,InferNoThrow,ResultsMatchDefault},
// smoke_CpuExecNetworkCheckMultiAppThreadSyncExecutionRejectsCpuPinning,
// smoke_CpuExecNetworkCheckMultiAppThreadSyncExecutionRejectsDefaultCpuPinning,
// smoke_CpuExecNetworkCheckMultiAppThreadSyncExecutionAllowedWithoutCpuPinning),
// so no replacement is added at the Config-unit-test level for the "effective value" semantics.
// This file is trimmed to what is still valid and meaningful at the `Config` level: parsing /
// validation of the raw `multiAppThreadSyncExecution` property and the legacy config-map
// shadowing check below.

TEST(MultiAppThreadSyncTest, LegacyConfigMapDoesNotShadowEffectiveValue) {
    // The compiled model resolves this property through an explicit handler that reports the
    // normalized value, so the legacy _config map must not carry a stale requested value.
    Config config;
    config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}});
    config.updateProperties();

    ASSERT_EQ(config._config.find(ov::intel_cpu::multi_app_thread_sync_execution.name()),
              config._config.end());
}

}  // namespace
