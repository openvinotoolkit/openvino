// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "config.h"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

using namespace ov::intel_cpu;

namespace {

// ---------------------------------------------------------------------------
// A1: Default value
// ---------------------------------------------------------------------------
TEST(MultiAppThreadSyncTest, Config_DefaultMultiAppThreadSyncIsFalse) {
    Config config;
    ASSERT_FALSE(config.multiAppThreadSyncExecution);
}

// ---------------------------------------------------------------------------
// A2: readProperties — set true
// ---------------------------------------------------------------------------
TEST(MultiAppThreadSyncTest, Config_ReadProperties_SetTrue) {
    Config config;
    OV_ASSERT_NO_THROW(config.readProperties(
        {{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}}));
    ASSERT_TRUE(config.multiAppThreadSyncExecution);
}

// ---------------------------------------------------------------------------
// A3: readProperties — toggle true → false
// ---------------------------------------------------------------------------
TEST(MultiAppThreadSyncTest, Config_ReadProperties_SetFalse) {
    Config config;
    config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}});
    ASSERT_TRUE(config.multiAppThreadSyncExecution);

    OV_ASSERT_NO_THROW(config.readProperties(
        {{ov::intel_cpu::multi_app_thread_sync_execution.name(), false}}));
    ASSERT_FALSE(config.multiAppThreadSyncExecution);
}

// ---------------------------------------------------------------------------
// A4: readProperties — invalid value throws
// ---------------------------------------------------------------------------
TEST(MultiAppThreadSyncTest, Config_ReadProperties_InvalidValueThrows) {
    Config config;
    ASSERT_THROW(
        config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(),
                                std::string("maybe")}}),
        ov::Exception);
}

// ---------------------------------------------------------------------------
// A5: updateProperties — true writes "YES" to _config map
// ---------------------------------------------------------------------------
TEST(MultiAppThreadSyncTest, Config_UpdateProperties_TrueWritesYES) {
    Config config;
    config.readProperties({{ov::intel_cpu::multi_app_thread_sync_execution.name(), true}});
    config.updateProperties();

    auto it = config._config.find(ov::intel_cpu::multi_app_thread_sync_execution.name());
    ASSERT_NE(it, config._config.end());
    ASSERT_EQ(it->second, "YES");
}

// ---------------------------------------------------------------------------
// A6: updateProperties — default (false) writes "NO" to _config map
// ---------------------------------------------------------------------------
TEST(MultiAppThreadSyncTest, Config_UpdateProperties_FalseWritesNO) {
    Config config;
    config.updateProperties();

    auto it = config._config.find(ov::intel_cpu::multi_app_thread_sync_execution.name());
    ASSERT_NE(it, config._config.end());
    ASSERT_EQ(it->second, "NO");
}

}  // namespace
