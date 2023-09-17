// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reverse_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U16,
        InferenceEngine::Precision::I32
};

const std::vector<int64_t> batchAxisIndices = { 0L };

const std::vector<int64_t> seqAxisIndices = { 1L };

const std::vector<std::vector<size_t>> inputShapes = { {3, 10} }; //, 10, 20

const std::vector<std::vector<size_t>> reversSeqLengthsVecShapes = { {3} };

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
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
