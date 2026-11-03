// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::test::snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;

class PassPipelineMergeTest : public testing::TestWithParam<bool> {};

TEST_P(PassPipelineMergeTest, PreservesBothConfigs) {
    PassPipeline lhs;
    lhs.register_pass<UpdateMemoryAccessCounts>(4);
    lhs.get_pass_config()->disable<UpdateMemoryAccessCounts>();
    lhs.get_pass_config()->enable<InitLoops>();

    PassPipeline rhs;
    rhs.register_pass<SetLoopIncrementOne>();
    rhs.get_pass_config()->disable<SetLoopIncrementOne>();
    rhs.get_pass_config()->enable<InsertLoadStore>();

    const auto merged = GetParam() ? PassPipeline::merge_pipelines(rhs, lhs) : PassPipeline::merge_pipelines(lhs, rhs);
    const auto& config = merged.get_pass_config();
    ASSERT_TRUE(config->is_disabled<UpdateMemoryAccessCounts>());
    ASSERT_TRUE(config->is_disabled<SetLoopIncrementOne>());
    EXPECT_TRUE(config->is_enabled<InitLoops>());
    EXPECT_TRUE(config->is_enabled<InsertLoadStore>());
    ASSERT_EQ(merged.get_passes().size(), 2);

    LinearIR linear_ir;
    EXPECT_NO_THROW(merged.run(linear_ir));
}

TEST_P(PassPipelineMergeTest, RejectsConflictingConfigs) {
    PassPipeline lhs;
    lhs.get_pass_config()->disable<SetLoopIncrementOne>();
    PassPipeline rhs;
    rhs.get_pass_config()->enable<SetLoopIncrementOne>();

    if (GetParam()) {
        EXPECT_THROW(PassPipeline::merge_pipelines(rhs, lhs), ov::AssertFailure);
    } else {
        EXPECT_THROW(PassPipeline::merge_pipelines(lhs, rhs), ov::AssertFailure);
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke, PassPipelineMergeTest, testing::Bool());

TEST(PassPipelineMerge, DoesNotModifyInputConfigs) {
    PassPipeline lhs;
    lhs.get_pass_config()->disable<SetLoopIncrementOne>();
    PassPipeline rhs;
    rhs.get_pass_config()->disable<SetLoopIncrementOne>();

    const auto merged = PassPipeline::merge_pipelines(lhs, rhs);
    EXPECT_TRUE(merged.get_pass_config()->is_disabled<SetLoopIncrementOne>());
    merged.get_pass_config()->enable<SetLoopIncrementOne>();
    EXPECT_TRUE(lhs.get_pass_config()->is_disabled<SetLoopIncrementOne>());
    EXPECT_TRUE(rhs.get_pass_config()->is_disabled<SetLoopIncrementOne>());
}

}  // namespace ov::test::snippets
