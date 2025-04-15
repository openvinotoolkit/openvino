// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<FullyConnectedShapes> shapes = {
    {
        { 1, 16 },
        { 16, 8 },
        false,
        false
    },
    {
        { 1, 16 },
        { 8, 16 },
        false,
        true
    },
    {
        { 16, 1 },
        { 16, 8 },
        true,
        false
    },
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
        ""
    },
    {
        false, // activation
        false, // per-channel
        true,  // FQ
        false, // bias
        ""
    },
    {
        true,  // activation
        true,  // per-channel
        false, // FQ
        false, // bias
        ""
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn({ov::element::i8/*, ov::element::u8*/}),
        ::testing::ValuesIn(activations),
        ::testing::Values("")),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace
