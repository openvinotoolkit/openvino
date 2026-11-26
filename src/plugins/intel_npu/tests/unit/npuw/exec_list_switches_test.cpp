// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>

#include "util.hpp"

using ov::npuw::util::exec_list_enabled_for;

namespace {

// The switches are read once per process, so the tests only hold with their defaults
bool switches_at_defaults() {
    for (const char* name : {"OPENVINO_NPUW_EXEC_LIST", "OPENVINO_NPUW_EXEC_LIST_GENERATE"}) {
        if (std::getenv(name) != nullptr) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST(NPUWExecListSwitchesTest, GqaStagesUseExecutionLists) {
    if (!switches_at_defaults()) {
        GTEST_SKIP() << "OPENVINO_NPUW_EXEC_LIST* is set";
    }
    EXPECT_TRUE(exec_list_enabled_for("model_gqa_prefill0"));
    EXPECT_TRUE(exec_list_enabled_for("model_gqa_generate1"));
}

TEST(NPUWExecListSwitchesTest, LlmPrefillAndGenerateVariantsUseExecutionLists) {
    if (!switches_at_defaults()) {
        GTEST_SKIP() << "OPENVINO_NPUW_EXEC_LIST* is set";
    }
    EXPECT_TRUE(exec_list_enabled_for("Model0_prefill"));
    EXPECT_TRUE(exec_list_enabled_for("Model0_kv1152"));
    EXPECT_TRUE(exec_list_enabled_for("Model0_kv128"));
}

TEST(NPUWExecListSwitchesTest, OtherModelsDoNotUseExecutionLists) {
    if (!switches_at_defaults()) {
        GTEST_SKIP() << "OPENVINO_NPUW_EXEC_LIST* is set";
    }
    EXPECT_FALSE(exec_list_enabled_for("Model0_lm_head"));
    EXPECT_FALSE(exec_list_enabled_for("Model0"));
    EXPECT_FALSE(exec_list_enabled_for("Model0_kv"));
    EXPECT_FALSE(exec_list_enabled_for("Model0_kv1152_extra"));
    EXPECT_FALSE(exec_list_enabled_for("Model0_prefill_extra"));
}

