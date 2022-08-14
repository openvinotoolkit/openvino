// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/einsum.hpp"

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<EinsumEquationWithInput> equationsWithInput = {
    { "ij->ji", {{{1, 2}}} }, // transpose 2d
    { "ijk->kij", { {1, 2, 3} } }, // transpose 3d
    { "ij->i", { {2, 3} } }, // reduce
    { "ab,cd->abcd", { { 1, 2}, {3, 4} } }, // no reduction
    { "ab,bc->ac", { {2, 3}, {3, 2} } }, // matrix multiplication
    { "ab,bcd,bc->ca", { {2, 4}, {4, 3, 1}, {4, 3} } },  // multiple multiplications
    { "kii->ki", { {1, 3, 3} } }, // diagonal
    { "abbac,bad->ad", { {2, 3, 3, 2, 4}, {3, 2, 1} } }, // diagonal and multiplication with repeated labels
    { "a...->...a", { {2, 2, 3} } }, // transpose with ellipsis
    { "a...->...", { {2, 2, 3} } }, // reduce with ellipsis
    { "ab...,...->ab...", { {2, 2, 3}, {1} } }, // multiply by scalar
    { "a...j,j...->a...", { {1, 1, 4, 3}, {3, 4, 2, 1} } } // complex multiplication
};

const auto params = ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(equationsWithInput),
        ::testing::Values(CommonTestUtils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_Einsum, EinsumLayerTest,
        params,
        EinsumLayerTest::getTestCaseName);

}  // namespace
