// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/paged_attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

namespace {
const std::vector<ov::AnyMap> additional_configs = {{{ov::intel_cpu::enable_sage_attn.name(), true}},
                                                    {{ov::intel_cpu::enable_sage_attn.name(), false}}};
const std::vector<InputShapes> inputShapeAndReorders = {  // greedy search
    {
        // L1, B, H, S
        {{-1, 1, 8, 64}, {{256, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {256, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSSDPATest,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(true, false),
                                            // TODO: Xattn should not direcctly compare with SDPA/decomposed Matmul
                                            // which not contain sparse logics
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::ValuesIn(additional_configs),
                                            ::testing::Values(false)),  // addSharedReader
                         PagedAttnTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSSDPATest_WithSlidingWindowAndSinks,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(false),        // extendBlockIndices
                                            ::testing::Values(false),        // enableXattn
                                            ::testing::Values(true, false),  // sinkInput
                                            ::testing::Values(0, 8),         // sliding_window = 8
                                            ::testing::Values(ov::AnyMap{
                                                {ov::intel_cpu::enable_sage_attn.name(), false}}),
                                            ::testing::Values(false)),  // addSharedReader
                         PagedAttnTestBase::getTestCaseName);

// PA1(write=true) + PA2(write=false) sharing the same KV cache.
// Verifies that PA2 reads the cache populated by PA1 and produces matching output.
INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnSharedKVCache,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(false),   // extendBlockIndices
                                            ::testing::Values(false),   // enableXattn
                                            ::testing::Values(false),   // sinkInput
                                            ::testing::Values(0),       // slidingWindow
                                            ::testing::Values(ov::AnyMap{
                                                {ov::intel_cpu::enable_sage_attn.name(), false}}),
                                            ::testing::Values(true)),   // addSharedReader
                         PagedAttnTestBase::getTestCaseName);

const std::vector<InputShapes> inputShapes = {  // greedy search
    {
        // L1, B, H, S
        {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSMatmulTest,
                         PagedAttnVSMatmulTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::f16),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),  // sinkInput = false
                                            ::testing::Values(0),      // sliding_window = 0
                                            ::testing::ValuesIn(additional_configs),
                                            ::testing::Values(false)),  // addSharedReader
                         PagedAttnTestBase::getTestCaseName);

const std::vector<InputShapes> fp16PrefillShapes = {{
                                                        {{-1, 1, 8, 64}, {{64, 1, 8, 64}, {1, 1, 8, 64}}},
                                                        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {64, 1, 8, 64}}},
                                                    },
                                                    {
                                                        {{-1, 1, 8, 64}, {{63, 1, 8, 64}, {1, 1, 8, 64}}},
                                                        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {63, 1, 8, 64}}},
                                                    },
                                                    {
                                                        // Exercise FP32 packed B's padding to a multiple of 32.
                                                        {{-1, 1, 8, 80}, {{63, 1, 8, 80}, {1, 1, 8, 80}}},
                                                        {{-1, 1, 8, 80}, {{0, 1, 8, 80}, {63, 1, 8, 80}}},
                                                    }};

class PagedAttnFP16PrefillTest : public PagedAttnVSMatmulTest {
public:
    void SetUp() override {
        PagedAttnTestBase::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto head_size = static_cast<int64_t>(targetStaticShapes[0][0][3]);
        function = get_model(ElementType::f16, false, head_size, 8, false, 0, false);
        functionRefs = get_ref_model(ElementType::f16, head_size, 8, false, false);
    }

    void generate(int idx,
                  bool isPagedAttn,
                  const std::vector<ov::Shape>& shapes,
                  bool extendBlockIndices,
                  bool useSinkInput) override {
        PagedAttnTestBase::generate(idx, isPagedAttn, shapes, extendBlockIndices, useSinkInput);
        // Keep FP16 QK finite and use 16 equally spaced levels representable by U4 and U8 caches.
        for (size_t port = 0; port < 3; ++port) {
            auto& tensor = inputs.at(function->get_parameters()[port]);
            auto* data = tensor.data<ov::float16>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = ov::float16(static_cast<float>(static_cast<int>((i + port * 3) % 16) - 8) / 32.f);
            }
        }
        if (!isPagedAttn) {
            auto& mask = inputs.at(function->get_parameters()[3]);
            std::fill_n(mask.data<ov::float16>(), mask.get_size(), ov::float16(0.f));
            auto& scale = inputs.at(function->get_parameters()[4]);
            scale.data<ov::float16>()[0] = ov::float16(1.f / std::sqrt(static_cast<float>(shapes[0][3])));
        }
    }
};

TEST_P(PagedAttnFP16PrefillTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto actual = run_test(function, false, false);
    past_len_count = 0;
    configuration[ov::hint::inference_precision.name()] = ov::element::f32;
    const auto expected = run_ref_test(functionRefs);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

// Exercise the FP16 BRGEMM path instead of promoting the attention inputs to FP32.
INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttnFP16Prefill,
    PagedAttnFP16PrefillTest,
    ::testing::Combine(::testing::Values(ElementType::f16),
                       ::testing::ValuesIn(fp16PrefillShapes),
                       ::testing::Values(false),
                       ::testing::Values(false),
                       ::testing::Values(false),
                       ::testing::Values(0),
                       ::testing::Values(ov::AnyMap{{ov::intel_cpu::enable_sage_attn.name(), false},
                                                    {ov::hint::inference_precision.name(), ov::element::f16},
                                                    {ov::hint::kv_cache_precision.name(), ov::element::f16}},
                                         ov::AnyMap{{ov::intel_cpu::enable_sage_attn.name(), false},
                                                    {ov::hint::inference_precision.name(), ov::element::f16},
                                                    {ov::hint::kv_cache_precision.name(), ov::element::u8}},
                                         ov::AnyMap{{ov::intel_cpu::enable_sage_attn.name(), false},
                                                    {ov::hint::inference_precision.name(), ov::element::f16},
                                                    {ov::hint::kv_cache_precision.name(), ov::element::u4}}),
                       ::testing::Values(false)),
    PagedAttnTestBase::getTestCaseName);
}  // namespace

TEST_P(PagedAttnCacheCollisionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP() << "PagedAttention requires AVX2 or higher for BRGEMM.";
    }

    // Run PA model (single compilation — both PA nodes share executor cache)
    past_len_count = 0;
    prepare();
    init_all_kv_caches(1024 / 32);
    std::vector<ov::Tensor> actualOutputs;
    int idx = 0;
    for (auto&& shapes : targetStaticShapes) {
        generate(idx++, true, shapes, false, false);
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto tensor = inferRequest.get_output_tensor(1);
        ov::Tensor copy{tensor.get_element_type(), tensor.get_shape()};
        tensor.copy_to(copy);
        actualOutputs.push_back(copy);
    }

    // Run SDPA reference for PA2 (head_size=hs2)
    past_len_count = 0;
    auto saved_function = function;
    function = functionRefs;
    prepare();
    std::vector<ov::Tensor> expectedOutputs;
    idx = 0;
    for (auto&& shapes : targetStaticShapes2_) {
        generate(idx++, false, shapes, false, false);
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto tensor = inferRequest.get_output_tensor(0);
        ov::Tensor copy{tensor.get_element_type(), tensor.get_shape()};
        tensor.copy_to(copy);
        expectedOutputs.push_back(copy);
    }
    reset();
    function = saved_function;

    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size());
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapes> inputShapesCacheCollision = {{
    // [L, B=1, H=4, S=256] — PA1 head_size; PA2 uses S=512
    {{-1, 1, 4, 256}, {{10, 1, 4, 256}, {1, 1, 4, 256}}},
    {{-1, 1, 4, 256}, {{0, 1, 4, 256}, {10, 1, 4, 256}}},
}};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnExecutorCacheCollision,
                         PagedAttnCacheCollisionTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapesCacheCollision),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(ov::AnyMap{
                                                {ov::intel_cpu::enable_sage_attn.name(), false}}),
                                            ::testing::Values(false)),
                         PagedAttnTestBase::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
