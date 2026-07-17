// Copyright (C)  2026 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "custom/subgraph_tests/src/classes/paged_attn.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

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

const std::vector<InputShapes> inputShapes = {  // greedy search
    {
        // L1, B, H, S
        {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSMatmulTest,
                         PagedAttnVSMatmulTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
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
