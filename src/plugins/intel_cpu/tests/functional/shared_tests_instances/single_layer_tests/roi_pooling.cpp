// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/roi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ROIPoolingLayerTest;

namespace {

const std::vector<ov::Shape> param_shapes = {
    {{1, 3, 8, 8}},
    {{3, 4, 50, 50}}
};

const std::vector<ov::Shape> coord_shapes = {
    {{1, 5}},
    {{3, 5}},
    {{5, 5}}
};

auto input_shapes = [](const std::vector<ov::Shape>& in1, const std::vector<ov::Shape>& in2) {
    std::vector<std::vector<ov::test::InputShape>> res;
    for (const auto& sh1 : in1)
        for (const auto& sh2 : in2)
            res.push_back(ov::test::static_shapes_to_test_representation({sh1, sh2}));
    return res;
}(param_shapes, coord_shapes);

const std::vector<ov::Shape> pooled_shapes_max = {
    {{1, 1}},
    {{2, 2}},
    {{3, 3}},
    {{6, 6}}
};

const std::vector<ov::Shape> pooled_shapes_bilinear = {
    {{1, 1}},
    {{2, 2}},
    {{3, 3}},
    {{6, 6}}
};

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
    ::testing::ValuesIn(input_shapes),
    ::testing::ValuesIn(pooled_shapes_max),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(ov::test::utils::ROIPoolingTypes::ROI_MAX),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_ROIPooling_bilinear = ::testing::Combine(
    ::testing::ValuesIn(input_shapes),
    ::testing::ValuesIn(pooled_shapes_bilinear),
    ::testing::Values(spatial_scales[1]),
    ::testing::Values(ov::test::utils::ROIPoolingTypes::ROI_BILINEAR),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest, test_ROIPooling_max, ROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTest, test_ROIPooling_bilinear, ROIPoolingLayerTest::getTestCaseName);

}  // namespace
