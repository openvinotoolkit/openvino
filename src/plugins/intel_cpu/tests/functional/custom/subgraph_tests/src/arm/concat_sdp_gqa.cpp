// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {
namespace {

// Grouped-query attention (several query heads per key/value head) through the stateful SDPA node, with the
// default u8 cache, plus an MHA control in the f16 suite. The grouped-query kernel paths are ARM64 only.
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

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ConcatSDPTest::generate_inputs(targetInputStaticShapes);
        if (!m_scale_query) {
            return;
        }
        // With q and k ~ N(0, 0.2), q.k / sqrt(S) has a deviation of 0.04: the softmax is nearly uniform and the
        // outputs hardly depend on the key scores. A 25 times larger query gives logits of unit deviation.
        auto& q = inputs.at(function->get_parameters()[0]);
        auto* data = q.data<float>();
        for (size_t i = 0; i < q.get_size(); ++i) {
            data[i] *= 25.0F;
        }
    }

    void run() override {
        // f16 kernels against the f32 ones on the same model and cache, where the cache quantization noise mostly
        // cancels, with key scores that matter; then against the decomposed reference with the base inputs, whose
        // larger tolerance covers the cache quantization.
        auto f32_config = configuration;
        f32_config[ov::hint::inference_precision.name()] = ov::element::f32;
        m_scale_query = true;
        const auto expected = run_test(function, f32_config);
        const auto actual = run_test(function, configuration);
        m_scale_query = false;
        const auto abs = abs_threshold;
        const auto rel = rel_threshold;
        abs_threshold = F16_VS_F32_ABS;
        rel_threshold = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            compare(expected[i], actual[i]);
        }
        abs_threshold = abs;
        rel_threshold = rel;
        ConcatSDPTest::run();
    }

    // About 10 f16 ulps for outputs in [1, 2) (the subgraph adds 1 to the attention). Largest difference measured over
    // all inference steps on Apple M5 with the scaled query: 4.9e-3 (GQA) and 4.3e-3 (MHA); breaking the f16 key loop
    // gives 0.16 to 0.24.
    static constexpr double F16_VS_F32_ABS = 1e-2;
    bool m_scale_query = false;
};

TEST_P(ConcatSDPGQAF16Test, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    // The stateful fusion must keep the cache in the node, as ConcatSDPTest checks, and the node must run in f16:
    // with an f32 fallback both runs above would use the same kernels.
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
    CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
    for (const auto& node : GetNodesWithTypes(compiledModel.get_runtime_model(), {"ScaledDotProductAttention"})) {
        EXPECT_EQ(node->get_rt_info().at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>(),
                  ov::element::f16);
    }
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

// MHA control: with one query head per key/value head, scores take dot_product_fp16<u8> (f16 products) and values
// accumulate per head, instead of the grouped paths; its NEON_FP16 loop runs on hosts without SVE. Greedy (B = 1,
// with a 20-token continuation) and beam search (B = 4).
INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPGQAF16Test_MHAControl,
                         ConcatSDPGQAF16Test,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(input_shapes[0], input_shapes[1]),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(8)),
                         ConcatSDPTest::getTestCaseName);

// Head size 80 in key and value quantization groups of 40. For the f16 value rows, per group: 2 steps of 16 and a
// tail of 8 with NEON_FP16 and 128-bit SVE, 1 step of 32 and a tail of 8 with 256-bit SVE, only the scalar loop at
// 512 bits; the key loops also run 2 steps of 16 and a tail of 8 per group. 6 query heads over 2 or 1 key/value heads
// give tiles of 3, and of 4 and 2, query heads; over 6, the per-head key path (dot_product_fp16<u8>).
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
                                            ::testing::Values(ov::AnyMap{{ov::value_cache_group_size.name(), "40"},
                                                                         {ov::key_cache_group_size.name(), "40"}}),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(6),
                                            ::testing::Values<int64_t>(6, 2, 1)),
                         ConcatSDPTest::getTestCaseName);

#endif  // OPENVINO_ARCH_ARM64

}  // namespace
}  // namespace test
}  // namespace ov
