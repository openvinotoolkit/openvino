// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "lpt_ngraph_functions/fuse_fake_quantize_and_scale_shift_function.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ngraph::builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
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
        ::testing::Values(ngraph::Shape({ 1, 3, 9, 9 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FuseFakeQuantizeAndScaleShiftTransformation::getTestCaseName);
}  // namespace
