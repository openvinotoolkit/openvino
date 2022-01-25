// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/dft.hpp>
#include <common_test_utils/test_constants.hpp>

namespace {

const std::initializer_list<ngraph::helpers::DFTOpType> opTypes {
    ngraph::helpers::DFTOpType::FORWARD,
//    ngraph::helpers::DFTOpType::INVERSE // TODO: idft is not implemented yet
};

const std::initializer_list<InferenceEngine::Precision> inputPrecision {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::initializer_list<std::vector<size_t>> inputShapes {
    {10, 4, 20, 32, 2},
    {2, 5, 7, 8, 2},
    {1, 120, 128, 1, 2},
};

// 1D DFT

const std::initializer_list<std::vector<int64_t>> axes1D {
    {0}, {1}, {2}, {3}, {-2}
};

const std::initializer_list<std::vector<int64_t>> signalSizes1D {
    {}, {16}, {40}
};

const auto combine = [](
    const std::initializer_list<std::vector<int64_t>> &axes,
    const std::initializer_list<std::vector<int64_t>> & signalSizes) {
    return testing::Combine(
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(inputPrecision),
        testing::ValuesIn(axes),
        testing::ValuesIn(signalSizes),
        testing::ValuesIn(opTypes),
        testing::Values(CommonTestUtils::DEVICE_GPU));
};

const auto testCase1D = combine(
    axes1D,
    signalSizes1D
);

// 2D DFT

const std::initializer_list<std::vector<int64_t>> axes2D {
    {0, 1}, {2, 1}, {2, 3}, {2, 0}, {1, 3}, {-1, -2}
};
const std::initializer_list<std::vector<int64_t>> signalSizes2D {
    {}, {5, 7}, {4, 10}, {16, 8}
};

const auto testCase2D = combine(
    axes2D,
    signalSizes2D
);

const auto smokeCase2d = testing::Combine(
    testing::Values(std::vector<size_t> {2, 3, 4, 1, 2}),
    testing::Values(InferenceEngine::Precision::FP32),
    testing::Values(std::vector<int64_t>{2, -4}),
    testing::Values(std::vector<int64_t>{5, 2}),
    testing::Values(ngraph::helpers::DFTOpType::FORWARD),
    testing::Values(CommonTestUtils::DEVICE_GPU)
);

// 3D DFT

const std::initializer_list<std::vector<int64_t>> axes3D {
    {0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {2, 3, 1}, {-3, -1, -2},
};

const std::initializer_list<std::vector<int64_t>> signalSizes3D {
    {}, {4, 8, 16}, {7, 11, 32}
};

const auto testCase3D = combine(
    axes3D,
    signalSizes3D
);

// 4D DFT

const std::initializer_list<std::vector<int64_t>> axes4D {
    {0, 1, 2, 3}, {-1, 2, 0, 1}
};

const std::initializer_list<std::vector<int64_t>> signalSizes4D {
    {}, {5, 2, 5, 2}
};

const auto testCase4D = combine(
    axes4D,
    signalSizes4D
);

using LayerTestsDefinitions::DFTLayerTest;

INSTANTIATE_TEST_SUITE_P(smoke_DFT_2d, DFTLayerTest, smokeCase2d, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DFT_1d, DFTLayerTest, testCase1D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DFT_2d, DFTLayerTest, testCase2D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DFT_3d, DFTLayerTest, testCase3D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DFT_4d, DFTLayerTest, testCase4D, DFTLayerTest::getTestCaseName);

} // namespace
