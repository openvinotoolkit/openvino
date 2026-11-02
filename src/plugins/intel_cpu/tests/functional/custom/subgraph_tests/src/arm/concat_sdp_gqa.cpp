// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {
namespace {

// Grouped-query attention (8 query heads over 2 or 1 key/value heads) through the stateful SDPA node, with the
// default u8 cache. The grouped-query kernel paths are ARM64 only.
#if defined(OPENVINO_ARCH_ARM64)

const std::vector<std::vector<InputShape>> input_shapes = {
    // greedy search, with a 20-token prompt continuation over a past
    {
        {{1, 8, -1, 64}, {{1, 8, 10, 64}, {1, 8, 1, 64}, {1, 8, 1, 64}, {1, 8, 20, 64}, {1, 8, 1, 64}}},
        {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 10, 64}, {1, 8, 11, 64}, {1, 8, 12, 64}, {1, 8, 32, 64}}},
    },
    // beam search
    {
        {{-1, 8, -1, 64}, {{4, 8, 10, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}}},
        {{-1, 8, -1, 64}, {{4, 8, 0, 64}, {4, 8, 10, 64}, {4, 8, 11, 64}, {4, 8, 12, 64}, {4, 8, 13, 64}}},
    },
    // big batch: 129 sequences, more than the threads of usual hosts
    {
        {{-1, 8, -1, 64}, {{129, 8, 10, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}}},
        {{-1, 8, -1, 64}, {{129, 8, 0, 64}, {129, 8, 10, 64}, {129, 8, 11, 64}, {129, 8, 12, 64}, {129, 8, 13, 64}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTest_GQA,
                         ConcatSDPTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(2, 1)),
                         ConcatSDPTest::getTestCaseName);

// The same subgraph with f16 inference precision on f32 inputs, so the node runs its f16 kernels. The f16
// instances of ConcatSDPTest itself are skipped on ARM.
class ConcatSDPGQAF16Test : public ConcatSDPTest {
protected:
    void SetUp() override {
        ConcatSDPTest::SetUp();
        if (!ov::with_cpu_neon_fp16()) {
            GTEST_SKIP() << "Host has no f16 vector arithmetic.";
        }
        configuration[ov::hint::inference_precision.name()] = ov::element::f16;
        // Both the node and the decomposed reference run in f16.
        abs_threshold = 0.03;
    }
};

TEST_P(ConcatSDPGQAF16Test, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPGQAF16Test,
                         ConcatSDPGQAF16Test,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(2)),
                         ConcatSDPTest::getTestCaseName);

#endif  // OPENVINO_ARCH_ARM64

}  // namespace
}  // namespace test
}  // namespace ov
