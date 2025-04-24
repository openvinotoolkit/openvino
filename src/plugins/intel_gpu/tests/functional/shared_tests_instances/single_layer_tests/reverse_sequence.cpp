// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reverse_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ReverseSequenceLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
        ov::element::i8,
        ov::element::u16,
        ov::element::i32
};

const std::vector<int64_t> batchAxisIndices = { 0L };

const std::vector<int64_t> seqAxisIndices = { 1L };

const std::vector<std::vector<size_t>> inputShapes = { {3, 10} }; //, 10, 20

const std::vector<std::vector<size_t>> reversSeqLengthsVecShapes = { {3} };

const std::vector<ov::test::utils::InputLayerType> secondaryInputTypes = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(Basic_smoke, ReverseSequenceLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(batchAxisIndices),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(inputShapes),
                            ::testing::ValuesIn(reversSeqLengthsVecShapes),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ReverseSequenceLayerTest::getTestCaseName);

}  // namespace
