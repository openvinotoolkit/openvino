// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_attention_token_type.hpp"

#include <cstdlib>

using namespace ov::test;
using namespace ov::op;

namespace ov {
namespace test {
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionTokenType,
                         PagedAttentionTokenTypeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(32),  // head_size
                                            ::testing::Values(1),   // head_num
                                            ::testing::Values(0),   // sliding_window_size
                                            ::testing::ValuesIn(PagedAttentionTokenTypeTest::GetTestDataForHeadSize32HeadNum1()),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PagedAttentionTokenTypeTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionTokenTypeWithSlidingWindow,
                         PagedAttentionTokenTypeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(32),  // head_size
                                            ::testing::Values(1),   // head_num
                                            ::testing::Values(5),   // sliding_window_size
                                            ::testing::ValuesIn(PagedAttentionTokenTypeTest::GetTestDataForHeadSize32HeadNum1SlidingWindowSize5()),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PagedAttentionTokenTypeTest::getTestCaseName);

class PagedAttentionTokenTypeTestNoFlashAttnV2 : public PagedAttentionTokenTypeTest {
protected:
    void SetUp() override {
        // Set env variable before compilation since this is a RELEASE_INTERNAL option
        setenv("OV_GPU_COULD_USE_FLASHATTN_V2", "0", 1);
        PagedAttentionTokenTypeTest::SetUp();
    }
    void TearDown() override {
        unsetenv("OV_GPU_COULD_USE_FLASHATTN_V2");
        PagedAttentionTokenTypeTest::TearDown();
    }
};

TEST_P(PagedAttentionTokenTypeTestNoFlashAttnV2, CompareWithPytorch) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionTokenType_NoFlashAttnV2,
                         PagedAttentionTokenTypeTestNoFlashAttnV2,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(32),  // head_size
                                            ::testing::Values(1),   // head_num
                                            ::testing::Values(0),   // sliding_window_size
                                            ::testing::ValuesIn(PagedAttentionTokenTypeTest::GetTestDataForHeadSize32HeadNum1()),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PagedAttentionTokenTypeTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionTokenTypeWithSlidingWindow_NoFlashAttnV2,
                         PagedAttentionTokenTypeTestNoFlashAttnV2,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(32),  // head_size
                                            ::testing::Values(1),   // head_num
                                            ::testing::Values(5),   // sliding_window_size
                                            ::testing::ValuesIn(PagedAttentionTokenTypeTest::GetTestDataForHeadSize32HeadNum1SlidingWindowSize5()),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PagedAttentionTokenTypeTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
