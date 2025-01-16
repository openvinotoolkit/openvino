// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/ctc_greedy_decoder_seq_len.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::CTCGreedyDecoderSeqLenLayerTest;

std::vector<std::vector<ov::Shape>> inputShape{{{1, 28, 41}}, {{1, 1, 1}}, {{1, 6, 10}}, {{3, 3, 16}}, {{5, 3, 55}}};

const std::vector<ov::element::Type> probPrecisions = {
    ov::element::f32,
    ov::element::f16
};
const std::vector<ov::element::Type> idxPrecisions = {
    ov::element::i32,
    ov::element::i64
};

std::vector<bool> mergeRepeated{true, false};

INSTANTIATE_TEST_SUITE_P(smoke_set1,
                         CTCGreedyDecoderSeqLenLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)),
                                            ::testing::Values(10),
                                            ::testing::ValuesIn(probPrecisions),
                                            ::testing::ValuesIn(idxPrecisions),
                                            ::testing::Values(0),
                                            ::testing::ValuesIn(mergeRepeated),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set2,
                         CTCGreedyDecoderSeqLenLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{{{2, 8, 11}},
                                                                                                 {{4, 10, 55}}})),
                                            ::testing::ValuesIn(std::vector<int>{5, 100}),
                                            ::testing::ValuesIn(probPrecisions),
                                            ::testing::ValuesIn(idxPrecisions),
                                            ::testing::ValuesIn(std::vector<int>{0, 5, 10}),
                                            ::testing::ValuesIn(mergeRepeated),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);
}  // namespace
