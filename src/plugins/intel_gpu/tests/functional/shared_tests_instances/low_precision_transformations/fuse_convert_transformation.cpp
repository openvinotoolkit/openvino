// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<element::Type> precisions = {
        element::f32,
        // element::f16 // TODO: temporarily commented due to failing in GPU Plugin on constant folding stage
};

const std::vector<ngraph::PartialShape>inputAndQuantizationShapes = {
        { 1, 4, 16, 16 },
};

const std::vector<ngraph::builder::subgraph::DequantizationOperations> deqOperations = {
        {
                { ngraph::element::f32 },
                {1.f},
                {0.45f}
        },
        {
                { ngraph::element::f32 },
                {},
                {0.45f}
        }
};

const std::vector<bool> constInput = { true, false };

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseConvertTransformation,
    ::testing::Combine(
            ::testing::ValuesIn(precisions),
            ::testing::ValuesIn(inputAndQuantizationShapes),
            ::testing::Values(ov::test::utils::DEVICE_GPU),
            ::testing::ValuesIn(deqOperations),
            ::testing::ValuesIn(constInput)),
    FuseConvertTransformation::getTestCaseName);
}  // namespace
