// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32
};

const std::vector<std::vector<std::vector<size_t>>> noneShapes = {
    {{1}, {1}, {1}},
    {{8}, {8}, {8}},
    {{4, 5}, {4, 5}, {4, 5}},
    {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}}
};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(noneShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
    ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

const std::vector<std::vector<std::vector<size_t>>> numpyShapes = {
    {{1}, {16}, {1}},
    {{1}, {1}, {16}},
    {{1}, {8}, {8}},
    {{8}, {1}, {8}},
    {{8}, {8}, {8}},
    {{4, 1}, {1}, {4, 8}},
    {{3, 8}, {8}, {3, 1}},
    {{8, 1}, {8, 1}, {8, 1}},
    {{1}, {5, 8}, {5, 8}},
    {{8, 1, 1}, {8, 1, 1}, {2, 5}},
    {{8, 1}, {6, 8, 1}, {6, 1, 1}},
    {{5, 1}, {8, 1, 7}, {5, 7}},
    {{2, 8, 1}, {2, 8, 9}, {2, 1, 9}},
    {{1, 4}, {8, 1, 1, 1}, {4}},
    {{5, 4, 1}, {8, 5, 1, 1}, {4, 1}},
    {{1, 4}, {6, 1, 8, 1}, {6, 1, 8, 4}},
    {{7, 3, 1, 8}, {7, 1, 1, 8}, {3, 2, 8}},
    {{1, 3, 1}, {8, 2, 3, 1}, {3, 9}},
    {{5, 1, 8}, {2, 1, 9, 8}, {2, 5, 9, 8}},
    {{6, 1, 1, 8}, {6, 7, 1, 8}, {2, 1}},
    {{5, 1, 1, 1}, {5, 7, 8, 6}, {1, 8, 6}}
};

const auto numpyCases = ::testing::Combine(
    ::testing::ValuesIn(numpyShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NUMPY),
    ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_CLDNN_TestsSelect_none, SelectLayerTest, noneCases, SelectLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_CLDNN_TestsSelect_numpy, SelectLayerTest, numpyCases, SelectLayerTest::getTestCaseName);
