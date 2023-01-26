// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::PartialShape>> inputShapes = {
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 1, 1}, {1, 128, 16, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 1, 1, 128}, {1, 128, 16, 64}},
        {{2, 68, 6, 92}, {2, 68, 6, 92}, {1, 1, 68, 68}, {2, 68, 6, 92}},
        {{1, 58, 16, 34}, {1, 58, 16, 34}, {1, 1, 1, 58}, {1, 58, 16, 34}},
};

static inline std::vector<std::vector<element::Type>> precisions() {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32, element::f32, element::f32},
    };
    // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
    if (InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16()) {
        prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16, element::bf16, element::bf16});
    }
    return prc;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHA,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShapes),
                             ::testing::ValuesIn(precisions()),
                             ::testing::ValuesIn({false, true}),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapeSelect = {
        // without broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 94, 12, 54}, {1, 94, 12, 54}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 94, 12, 54}},
        // with broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}},
        {{2, 52, 6, 102}, {2, 52, 6, 102}, {1, 6, 52, 52}, {1, 6, 1, 1}, {1, 6, 1, 1}, {2, 52, 6, 102}}
};

static inline std::vector<std::vector<element::Type>> precisionsSelect() {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32, element::f32, element::f32, element::f32, element::f32},
    };
    // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
    if (InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16()) {
        prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16, element::bf16, element::bf16, element::bf16, element::bf16});
    }
    return prc;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHASelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeSelect),
                                 ::testing::ValuesIn(precisionsSelect()),
                                 ::testing::Values(false),  // Need to support True for graph builder in tests
                                 ::testing::Values(2), // Less + MHA
                                 ::testing::Values(2),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesWOTranspose = {
        {{1, 12, 197, 64}, {1, 12, 64, 197}, {1, 12, 197, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeOnInputs, MHAWOTransposeOnInputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose),
                                 ::testing::Values(std::vector<ov::element::Type>{}),
                                 ::testing::Values(true),  // Need to support False for graph builder in tests
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAINT8MatMul, MHAINT8MatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<std::vector<ov::PartialShape>>(inputShapes.begin(), inputShapes.begin() + 2)),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(6),     // FQx3 on inputs + MHA + Transpose on output + Deq Mul
                                 ::testing::Values(5),     // FQx3 on inputs + MHA + Deq Mul
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAFQAfterMatMul, MHAFQAfterMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(3),     // MHA + Transpose on output + Deq Mul
                                 ::testing::Values(2),     // MHA + Deq Mul
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAFQ, MHAFQ,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 64, 12, 64}, {1, 64, 12, 64}, {1, 1, 1, 64}, {1, 64, 12, 64}}),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(7),     // Transposex2 + Subgraphsx5
                                 ::testing::Values(5),     // MHA + Deq Mul on output + Deqs on inputs + 2 xFQ on inputs
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov