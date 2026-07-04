// Copyright (C)  2026 FUJITSU LIMITED
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
#include "utils/arm_isa_support.h"
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
    if (inType == ElementType::f16)
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
const std::vector<ov::AnyMap> additional_configs = {
        {
            {ov::intel_cpu::enable_sage_attn.name(), false},
            {ov::hint::kv_cache_precision.name(), ov::element::f32},
            {ov::key_cache_precision.name(), ov::element::f32},
            {ov::value_cache_precision.name(), ov::element::f32},
        }, 
        {
            {ov::intel_cpu::enable_sage_attn.name(), false},
            {ov::hint::kv_cache_precision.name(), ov::element::u8},
            {ov::key_cache_precision.name(), ov::element::u8},
            {ov::value_cache_precision.name(), ov::element::u8},
        }
    };
const std::vector<InputShapes> inputShapeAndReorders = {  // greedy search
    {
        // L1, B, H, S
        {{-1, 1, 8, 64}, {{256, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {256, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSSDPATest,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
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
                         ::testing::Combine(::testing::Values(ElementType::f32),
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
    // reference model does not implement f16, hence skip the test for now
    if (inType == ElementType::f16) {
        GTEST_SKIP();
    }
    // If not SVE machine skip the test
    if (!ov::intel_cpu::hasArmISASupport(ov::intel_cpu::ArmISA::SVE)) {
        GTEST_SKIP();
    }
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
                                            ::testing::Values(false),
                                            ::testing::Values(false),          // enableXatten = false 
                                            ::testing::Values(true, false),    // sinkInput = true/false
                                            ::testing::Values(0),              // sliding_window = 0
                                            ::testing::ValuesIn(additional_configs),
                                            ::testing::Values(false)),         // addSharedReader
                         PagedAttnTestBase::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
