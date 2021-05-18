// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/mvn_multiply_add.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecision = {
        InferenceEngine::Precision::FP32
};

std::vector<InferenceEngine::Precision> idxPrecision = {
        InferenceEngine::Precision::I64
};

const std::vector<bool> acrossChannels = {
        true,
        false
};

const std::vector<bool> normalizeVariance = {
        true,
        false
};

const std::vector<float> epsilon = {
        0.000000001
};

const std::vector<std::string> epsMode = {
        "inside_sqrt",
        "outside_sqrt"
};

const std::vector<std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector>> shapes_1D = {
        { InferenceEngine::SizeVector{5}, InferenceEngine::SizeVector{5}},
        { InferenceEngine::SizeVector{64}, InferenceEngine::SizeVector{64}},
};

const std::vector<std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector>> shapes_2D = {
        { InferenceEngine::SizeVector{1, 5}, InferenceEngine::SizeVector{1, 5}},
        { InferenceEngine::SizeVector{2, 17}, InferenceEngine::SizeVector{1, 17}},
        { InferenceEngine::SizeVector{9, 64}, InferenceEngine::SizeVector{1, 64}},
        { InferenceEngine::SizeVector{5, 15}, InferenceEngine::SizeVector{1, 15}},
};

const std::vector<std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector>> shapes_3D = {
        { InferenceEngine::SizeVector{1, 5, 8}, InferenceEngine::SizeVector{1, 5, 8}},
        { InferenceEngine::SizeVector{2, 17, 9}, InferenceEngine::SizeVector{1, 1, 9}},
        { InferenceEngine::SizeVector{1, 1, 10}, InferenceEngine::SizeVector{1, 1, 10}},
        { InferenceEngine::SizeVector{2, 3, 3}, InferenceEngine::SizeVector{2, 3, 3}},
};

INSTANTIATE_TEST_CASE_P(smoke_MVNMultiplyAdd_1D, MVNMultiplyAdd,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes_1D),
                                ::testing::ValuesIn(netPrecision),
                                ::testing::ValuesIn(idxPrecision),
                                ::testing::Values(std::vector<int>{0}),
                                ::testing::ValuesIn(normalizeVariance),
                                ::testing::ValuesIn(epsilon),
                                ::testing::ValuesIn(epsMode),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MVNMultiplyAdd::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MVNMultiplyAdd_2D, MVNMultiplyAdd,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes_2D),
                                ::testing::ValuesIn(netPrecision),
                                ::testing::ValuesIn(idxPrecision),
                                ::testing::Values(std::vector<int>{1}),
                                ::testing::ValuesIn(normalizeVariance),
                                ::testing::ValuesIn(epsilon),
                                ::testing::ValuesIn(epsMode),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MVNMultiplyAdd::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MVNMultiplyAdd_3D, MVNMultiplyAdd,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes_3D),
                                ::testing::ValuesIn(netPrecision),
                                ::testing::ValuesIn(idxPrecision),
                                ::testing::Values(std::vector<int>{2}),
                                ::testing::ValuesIn(normalizeVariance),
                                ::testing::ValuesIn(epsilon),
                                ::testing::ValuesIn(epsMode),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MVNMultiplyAdd::getTestCaseName);

}  // namespace
