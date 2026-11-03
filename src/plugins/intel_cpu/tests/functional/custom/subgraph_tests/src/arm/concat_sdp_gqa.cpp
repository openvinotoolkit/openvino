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
    // big batch: 129 sequences, at least the thread count of usual hosts, for the per-sequence value path
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

// The same subgraph with f16 inference precision on f32 inputs, so the node runs its f16 kernels. ConcatSDPTest's
// own f16 instances skip without x86 AVX512-FP16, so on every ARM host.
class ConcatSDPGQAF16Test : public ConcatSDPTest {
protected:
    void SetUp() override {
        if (!ov::with_cpu_neon_fp16()) {
            GTEST_SKIP() << "Host has no f16 vector arithmetic.";
        }
        ConcatSDPTest::SetUp();
        configuration[ov::hint::inference_precision.name()] = ov::element::f16;
        // Both the node and the decomposed reference run in f16.
        abs_threshold = 0.03;
    }

    void run() override {
        ConcatSDPTest::run();
        // The same model and cache with f32 kernels: the cache quantization noise cancels, so a tighter bound
        // checks the f16 kernels against the f32 ones.
        auto f32_config = configuration;
        f32_config[ov::hint::inference_precision.name()] = ov::element::f32;
        const auto expected = run_test(function, f32_config);
        const auto actual = run_test(function, configuration);
        abs_threshold = F16_VS_F32_ABS;
        rel_threshold = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            compare(expected[i], actual[i]);
        }
    }

    // About 2 f16 ulps at the output magnitude (~1.2), and over 5 times the largest difference measured.
    static constexpr double F16_VS_F32_ABS = 2e-3;
};

TEST_P(ConcatSDPGQAF16Test, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    // The stateful fusion must keep the cache in the node, as ConcatSDPTest checks.
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
    CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
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

// Head size 80 in value quantization groups of 40, for the f16 value rows: per group, 2 steps of 16 and a tail of 8
// with NEON_FP16 and 128-bit SVE, 1 step of 32 and a tail of 8 with 256-bit SVE, only the scalar loop at 512 bits.
// 6 query heads over 2 or 1 key/value heads give tiles of 3, and of 4 and 2, query heads.
const std::vector<std::vector<InputShape>> input_shapes_head80 = {
    {
        {{-1, 6, -1, 80}, {{4, 6, 10, 80}, {4, 6, 1, 80}, {4, 6, 1, 80}, {4, 6, 1, 80}, {4, 6, 1, 80}}},
        {{-1, 6, -1, 80}, {{4, 6, 0, 80}, {4, 6, 10, 80}, {4, 6, 11, 80}, {4, 6, 12, 80}, {4, 6, 13, 80}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPGQAF16Test_ValueGroups,
                         ConcatSDPGQAF16Test,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_head80),
                                            ::testing::Values(ov::AnyMap{{ov::value_cache_group_size.name(), "40"}}),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(6),
                                            ::testing::Values<int64_t>(2, 1)),
                         ConcatSDPTest::getTestCaseName);

#endif  // OPENVINO_ARCH_ARM64

}  // namespace
}  // namespace test
}  // namespace ov
