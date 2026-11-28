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

// Effective "is sync-in-caller-thread actually safe" semantics now live in CompiledModel (which
// also sees sub-compiled-models/exclusiveAsyncRequests), not Config; covered end-to-end by the
// functional tests in ov_executable_network/properties.cpp. This file only covers Config-level
// parsing/validation.

TEST(MultiAppThreadSyncTest, LegacyConfigMapDoesNotShadowEffectiveValue) {
    // CompiledModel resolves this property via an explicit handler, so the legacy _config map
    // must not carry a stale requested value.
    Config config;
    config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}});
    config.updateProperties();

    ASSERT_EQ(config._config.find(ov::intel_cpu::multi_app_thread_sync_execution.name()),
              config._config.end());
}

}  // namespace
