// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_precision_selection_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ov_lpt_models/fake_quantize.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<FakeQuantizePrecisionSelectionTransformationTestValues> testValues = {
    {
        { ov::element::u8, ov::element::i8 },
        { ov::element::u8 },
        true,
        {
            { 256ul, { }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } }
        },
        {
            ov::element::u8,
            { 256ul, { }, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
            { }
        },
    },
    {
        { ov::element::u8, ov::element::i8 },
        { ov::element::i8 }, // Convolution on CPU doesn't support it, but it will be not used
        // INT8 is not available for limited operation (Convolution)
        false,
        {
            { 256ul, { }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } }
        },
        {
            // original precision is used
            ov::element::u8,
            // FakeQuantize has to select the first available: U8, not limited operation required I8 but this fact doesn't affect
            { 256ul, { }, { 0.f }, { 25.5f }, { 0.f }, { 255.f } },
            // FakeQuantize on weights is not changed
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } }
        },
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FakeQuantizePrecisionSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 32, 72, 48 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(testValues)),
    FakeQuantizePrecisionSelectionTransformation::getTestCaseName);
}  // namespace
