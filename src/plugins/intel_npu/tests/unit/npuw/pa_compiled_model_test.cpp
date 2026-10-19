// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

using ov::npuw::PACompiledModel;

std::string str(const ov::AnyMap& config, const char* key) {
    return config.at(key).as<std::string>();
}

TEST(PAVariantConfig, PartitionsThePAOpsOffNPU) {
    const auto config = PACompiledModel::make_variant_config({}, "CPU", "bank0");

    EXPECT_EQ(str(config, "NPU_USE_NPUW"), "YES");
    EXPECT_EQ(str(config, "NPUW_DEVICES"), "NPU,CPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_PIPELINE"), "REP");
    EXPECT_EQ(str(config, "NPUW_ONLINE_ISOLATE"), "Op:PagedAttentionExtension/pa");
    EXPECT_EQ(str(config, "NPUW_ONLINE_AVOID"), "Op:PagedAttentionExtension/NPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_KEEP_BLOCKS_TAGGED"), "pa");
    EXPECT_EQ(str(config, "NPUW_FOLD"), "YES");
    EXPECT_EQ(str(config, "NPUW_WEIGHTS_BANK"), "bank0");
}

TEST(PAVariantConfig, ForwardsTheCallerPropertiesButNotTheWholeModelChoices) {
    const ov::AnyMap properties = {{"NPUW_PA", "YES"},
                                   {"NPUW_PLAN", "plan.xml"},
                                   {"NPUW_SUBMODEL_DEVICE", "0:CPU"},
                                   {"NPUW_DUMP_SUBS", "YES"},
                                   {"KV_CACHE_PRECISION", "f16"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank1");

    EXPECT_EQ(config.count("NPUW_PA"), 0u);
    EXPECT_EQ(config.count("NPUW_PLAN"), 0u);
    EXPECT_EQ(config.count("NPUW_SUBMODEL_DEVICE"), 0u);
    EXPECT_EQ(str(config, "NPUW_DUMP_SUBS"), "YES");
    EXPECT_EQ(str(config, "KV_CACHE_PRECISION"), "f16");
}

TEST(PAVariantConfig, CallerPropertiesOverrideTheDefaults) {
    const ov::AnyMap properties = {{"NPUW_DEVICES", "CPU"},
                                   {"NPUW_ONLINE_AVOID", "Op:Select/NPU"},
                                   {"NPU_USE_NPUW", "NO"},
                                   {"NPUW_WEIGHTS_BANK", "caller"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank2");

    EXPECT_EQ(str(config, "NPUW_DEVICES"), "CPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_AVOID"), "Op:Select/NPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_ISOLATE"), "Op:PagedAttentionExtension/pa");
    // A variant is always an NPUW model with the bank the front-end chose.
    EXPECT_EQ(str(config, "NPU_USE_NPUW"), "YES");
    EXPECT_EQ(str(config, "NPUW_WEIGHTS_BANK"), "bank2");
}

}  // namespace
