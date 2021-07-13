// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/mvn_fq_mvn.hpp"

using namespace SubgraphTestsDefinitions;
using namespace InferenceEngine;

namespace {

const std::vector<Precision> netPrecision = {
        Precision::FP32
};

std::vector<Precision> idxPrecision = {
        Precision::I64
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

const std::vector<size_t> level = {255};

const std::vector<std::vector<size_t>> constShapes = {
        {1, 1, 1, 1},
        {1, 5, 1, 1}
};

const std::vector<std::vector<float>> inputParams = {
        {-10, 10, 0.2},
        {0, 10, 0.2}
};

const auto fqParams = ::testing::Combine(
        ::testing::ValuesIn(level),
        ::testing::ValuesIn(constShapes),
        ::testing::ValuesIn(inputParams)
);

const std::vector<SizeVector> dataShapes = {
        {1, 5, 1, 1},
        {1, 5, 1, 2},
        {1, 5, 1, 3},
        {1, 5, 1, 4},
        {1, 5, 1, 5},
        {1, 5, 1, 6},
        {1, 5, 1, 7},
        {1, 5, 1, 8},
        {1, 5, 1, 9},
        {1, 5, 1, 10},
        {1, 5, 1, 11},
        {1, 5, 1, 12},
        {1, 5, 1, 13},
        {1, 5, 1, 14},
        {1, 5, 1, 15},
        {1, 5, 1, 16}
};

INSTANTIATE_TEST_CASE_P(smoke_MVNFqMVN, MvnFqMvnSubgraphTest,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(dataShapes),
                                ::testing::ValuesIn(netPrecision),
                                ::testing::ValuesIn(idxPrecision),
                                ::testing::Values(std::vector<int>{2, 3}),
                                ::testing::ValuesIn(normalizeVariance),
                                ::testing::ValuesIn(epsilon),
                                ::testing::ValuesIn(epsMode),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MvnFqMvnSubgraphTest::getTestCaseName);
}  // namespace
