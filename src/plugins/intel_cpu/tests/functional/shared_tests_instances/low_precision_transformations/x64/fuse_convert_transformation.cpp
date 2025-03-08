// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector< ov::PartialShape > inputAndQuantizationShapes = {
        { 1, 4, 16, 16 },
};

const std::vector<ov::builder::subgraph::DequantizationOperations> deqOperations = {
        {
                { ov::element::f32 },
                {1.f},
                {0.45f}
        },
        {
                { ov::element::f32 },
                {},
                {0.45f}
        }
};

const std::vector<bool> constInput = { true, false };

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseConvertTransformation,
    ::testing::Combine(
            ::testing::ValuesIn(precisions),
            ::testing::ValuesIn(inputAndQuantizationShapes),
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::ValuesIn(deqOperations),
            ::testing::ValuesIn(constInput)),
    FuseConvertTransformation::getTestCaseName);
}  // namespace
