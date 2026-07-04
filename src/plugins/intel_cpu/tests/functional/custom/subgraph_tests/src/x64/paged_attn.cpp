// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"
#include "custom/subgraph_tests/src/classes/paged_attn.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

TEST_P(PagedAttnVSSDPATest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, additional_config,
                 addSharedReader] = this->GetParam();
    const bool isSageAttn =
        intel_cpu::contains_key_value(additional_config, {ov::intel_cpu::enable_sage_attn.name(), true});
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (isSageAttn && !(ov::with_cpu_x86_avx512_core_amx_int8() || CPUTestUtils::with_cpu_x86_avx2_vnni_2()))
        GTEST_SKIP();

    past_len_count = 0;

    // compare the logits from paged attn and sdpa
    auto actualOutputs = run_test(function, extendBlockIndices, sinkInput);
    // reference model doesn't support sage attention
    if (isSageAttn) {
        configuration[ov::intel_cpu::enable_sage_attn.name()] = false;
    }
    // Reset past_len_count before running reference test to ensure consistent mask generation
    past_len_count = 0;
    auto expectedOutputs = run_ref_test(functionRefs, sinkInput);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

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
}  // namespace

TEST_P(PagedAttnVSMatmulTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType,
                 inputShapes,
                 extendBlockIndices,
                 enableXattn,
                 sinkInput,
                 slidingWindow,
                 additional_config,
                 addSharedReader] = this->GetParam();
    ASSERT_FALSE(addSharedReader) << "PagedAttnVSMatmulTest does not support shared KV-cache (addSharedReader=true)";
    const bool isSageAttn =
        intel_cpu::contains_key_value(additional_config, {ov::intel_cpu::enable_sage_attn.name(), true});
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (isSageAttn && !(ov::with_cpu_x86_avx512_core_amx_int8() || CPUTestUtils::with_cpu_x86_avx2_vnni_2()))
        GTEST_SKIP();
    // compare the logits from paged attn and sdpa
    auto actualOutputs = run_test(function, extendBlockIndices, false);
    // reference model doesn't support sage attention, disable it
    if (isSageAttn) {
        configuration[ov::intel_cpu::enable_sage_attn.name()] = false;
    }
    auto expectedOutputs = run_ref_test(functionRefs);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {

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
}  // namespace

TEST_P(PagedAttnCacheCollisionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

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
