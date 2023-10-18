// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/roi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ROIPoolingLayerTest;

const std::vector<ov::Shape> inShapes = {
    {{1, 3, 8, 8}},
    {{3, 4, 50, 50}},
};

const std::vector<ov::Shape> pooledShapes_max = {
    {{1, 1}},
    {{2, 2}},
    {{3, 3}},
    {{6, 6}},
};

const std::vector<ov::Shape> pooledShapes_bilinear = {
    {{1, 1}},
    {{2, 2}},
    {{3, 3}},
    {{6, 6}},
};

const std::vector<ov::Shape> coordShapes = {
    {{1, 5}},
    {{3, 5}},
    {{5, 5}},
};

const std::vector<ov::element::Type> netPRCs = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

auto input_shapes = [](const std::vector<ov::Shape>& in1, const std::vector<ov::Shape>& in2) {
    std::vector<std::vector<ov::test::InputShape>> res;
    for (const auto& sh1 : in1)
        for (const auto& sh2 : in2)
            res.push_back(ov::test::static_shapes_to_test_representation({sh1, sh2}));
    return res;
}(inShapes, coordShapes);

const auto params_max = testing::Combine(testing::ValuesIn(input_shapes),
                                         testing::ValuesIn(pooledShapes_max),
                                         testing::ValuesIn(spatial_scales),
                                         testing::Values(ov::test::utils::ROIPoolingTypes::ROI_MAX),
                                         testing::ValuesIn(netPRCs),
                                         testing::Values(ov::test::utils::DEVICE_GPU));

const auto params_bilinear = testing::Combine(testing::ValuesIn(input_shapes),
                                              testing::ValuesIn(pooledShapes_bilinear),
                                              testing::Values(spatial_scales[1]),
                                              testing::Values(ov::test::utils::ROIPoolingTypes::ROI_BILINEAR),
                                              testing::ValuesIn(netPRCs),
                                              testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_ROIPooling_max,
                         ROIPoolingLayerTest,
                         params_max,
                         ROIPoolingLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIPooling_bilinear,
                         ROIPoolingLayerTest,
                         params_bilinear,
                         ROIPoolingLayerTest::getTestCaseName);

}  // namespace
