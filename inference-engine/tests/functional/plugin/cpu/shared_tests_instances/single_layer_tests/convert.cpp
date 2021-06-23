// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convert.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace {
const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<Precision> precisions = {
        Precision::U8,
        Precision::I8,
        Precision::U16,
        Precision::I16,
        Precision::I32,
        Precision::U64,
        Precision::I64,
        Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertLayerTest, ConvertLayerTest,
                        ::testing::Combine(
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvertLayerTest::getTestCaseName);

}  // namespace