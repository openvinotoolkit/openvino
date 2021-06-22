// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/mvn_multiply_add.hpp"

using namespace SubgraphTestsDefinitions;
using namespace InferenceEngine;

namespace {

const std::vector<Precision> netPrecision = {
        Precision::FP32
};

std::vector<Precision> idxPrecision = {
        Precision::I64
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

const std::vector<std::pair<SizeVector, SizeVector>> shapes_1D = {
        std::pair<SizeVector, SizeVector>{ SizeVector{5}, SizeVector{5}},
        std::pair<SizeVector, SizeVector>{ SizeVector{64}, SizeVector{64}},
};

const std::vector<std::pair<SizeVector, SizeVector>> shapes_2D = {
        std::pair<SizeVector, SizeVector>{ SizeVector{1, 5}, SizeVector{1, 5}},
        std::pair<SizeVector, SizeVector>{ SizeVector{2, 17}, SizeVector{1, 17}},
        std::pair<SizeVector, SizeVector>{ SizeVector{9, 64}, SizeVector{1, 64}},
        std::pair<SizeVector, SizeVector>{ SizeVector{5, 15}, SizeVector{1, 15}},
};

const std::vector<std::pair<SizeVector, SizeVector>> shapes_3D = {
        std::pair<SizeVector, SizeVector>{ SizeVector{1, 5, 8}, SizeVector{1, 5, 8}},
        std::pair<SizeVector, SizeVector>{ SizeVector{2, 17, 9}, SizeVector{1, 1, 9}},
        std::pair<SizeVector, SizeVector>{ SizeVector{1, 1, 10}, SizeVector{1, 1, 10}},
        std::pair<SizeVector, SizeVector>{ SizeVector{2, 3, 3}, SizeVector{2, 3, 3}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MVNMultiplyAdd_1D, MVNMultiplyAdd,
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

INSTANTIATE_TEST_SUITE_P(smoke_MVNMultiplyAdd_2D, MVNMultiplyAdd,
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

INSTANTIATE_TEST_SUITE_P(smoke_MVNMultiplyAdd_3D, MVNMultiplyAdd,
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
