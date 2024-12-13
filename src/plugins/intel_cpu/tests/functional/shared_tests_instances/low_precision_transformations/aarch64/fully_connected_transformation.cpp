// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<FullyConnectedShapes> shapes = {
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 16, 8 },
        false,
        false
    },
    {
        ov::PartialShape{ 1, 1, 16 },
        ov::PartialShape{ 1, 16, 8 },
        false,
        false
    },
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 8, 16 },
        false,
        true
    },
    {
        ov::PartialShape{ 1, 1, 16 },
        ov::PartialShape{ 1, 8, 16 },
        false,
        true
    },
    {
        ov::PartialShape{ 16, 1 },
        ov::PartialShape{ 16, 8 },
        true,
        false
    },
    {
        ov::PartialShape{ 1, 16, 1 },
        ov::PartialShape{ 1, 16, 8 },
        true,
        false
    },
    {
        ov::PartialShape{ 16, 1 },
        ov::PartialShape{ 8, 16 },
        true,
        true
    },
    {
        ov::PartialShape{ 1, 16, 1 },
        ov::PartialShape{ 1, 8, 16 },
        true,
        true
    }
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<FullyConnectedParams> activations = {
    {
        true,  // activation
        false, // per-channel
        true,  // FQ
        false, // bias
        "fullyConnected,relu_original"
    },
    {
        false,  // activation
        false,  // per-channel
        true,   // FQ
        false,  // bias
        "fullyconnected_original"
    },
    {
        true,  // activation
        true,  // per-channel
        false, // FQ
        false, // bias
        "fullyConnected,relu_original" // dequantization is not supported for per-channel quantization
    },
    {
        true,  // activation
        false, // per-channel
        true,  // FQ
        true,  // bias
        "fullyConnected"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn({ov::element::i8 /*, ov::element::u8*/}),
        ::testing::ValuesIn(activations),
        ::testing::Values("gemm_acl_i8")),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace
