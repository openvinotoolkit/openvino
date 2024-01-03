// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
        ngraph::element::f32,
        // ngraph::element::f16
};

const std::vector<MatMulShapes> shapes = {
    {
        ngraph::PartialShape{ 1, 16 },
        ngraph::PartialShape{ 16, 8 },
        false,
        false
    },
    {
        ngraph::PartialShape{ 1, 16 },
        ngraph::PartialShape{ 8, 16 },
        false,
        true
    },
    {
        ngraph::PartialShape{ 16, 1 },
        ngraph::PartialShape{ 16, 8 },
        true,
        false
    },
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace
