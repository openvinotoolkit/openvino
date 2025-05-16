// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_fill_empty_rows.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test::SparseFillEmptyRows {
INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsF32I32, SparseFillEmptyRowsLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SparseFillEmptyRowsParamsVector),
                        ::testing::ValuesIn(std::vector<ElementType>{ElementType::f32, ElementType::f16}),
                        ::testing::Values(ElementType::i32),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_f32"})),
                SparseFillEmptyRowsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsI32I32, SparseFillEmptyRowsLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SparseFillEmptyRowsParamsVector),
                        ::testing::Values(ElementType::i32),
                        ::testing::Values(ElementType::i32),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i32"})),
                SparseFillEmptyRowsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsUI8I32, SparseFillEmptyRowsLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SparseFillEmptyRowsParamsVector),
                        ::testing::Values(ElementType::u8, ElementType::i8),
                        ::testing::Values(ElementType::i32),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i8"})),
                SparseFillEmptyRowsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsF32I64, SparseFillEmptyRowsLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SparseFillEmptyRowsParamsVector),
                        ::testing::ValuesIn(std::vector<ElementType>{ElementType::f32, ElementType::f16}),
                        ::testing::Values(ElementType::i64),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_f32"})),
                SparseFillEmptyRowsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsI32I64, SparseFillEmptyRowsLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SparseFillEmptyRowsParamsVector),
                        ::testing::Values(ElementType::i32),
                        ::testing::Values(ElementType::i64),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i32"})),
                SparseFillEmptyRowsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsUI8I64, SparseFillEmptyRowsLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SparseFillEmptyRowsParamsVector),
                        ::testing::Values(ElementType::u8, ElementType::i8),
                        ::testing::Values(ElementType::i64),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i8"})),
                SparseFillEmptyRowsLayerCPUTest::getTestCaseName);

}  // namespace ov::test::SparseFillEmptyRows
