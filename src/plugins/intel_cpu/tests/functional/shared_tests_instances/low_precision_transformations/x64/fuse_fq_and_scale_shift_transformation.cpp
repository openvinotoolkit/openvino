// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ov_lpt_models/fuse_fake_quantize_and_scale_shift.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ov::builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
// TODO: Issue 39810
//    {
//        256ul,
//        { 1ul, 3ul, 1ul, 1ul },
//        { 0.f, 0.f, 0.f },
//        { 2.55f / 10.f, 2.55f / 5.f, 2.55f / 2.f },
//        { 0.f, 0.f, 0.f },
//        { 2.55f / 10.f, 2.55f / 5.f, 2.55f / 2.f }
//    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseFakeQuantizeAndScaleShiftTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 9, 9 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FuseFakeQuantizeAndScaleShiftTransformation::getTestCaseName);
}  // namespace
