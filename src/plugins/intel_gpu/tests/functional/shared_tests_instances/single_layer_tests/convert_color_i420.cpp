// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/convert_color_i420.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConvertColorI420LayerTest;

const std::vector<ov::Shape> inShapes_nhwc = {
    {1, 10, 10, 1}
};

const std::vector<ov::element::Type> inTypes = {
    ov::element::u8,
    ov::element::f32
};

auto generate_input_static_shapes = [] (const std::vector<ov::Shape>& original_shapes, bool single_plane) {
    std::vector<std::vector<ov::Shape>> result_shapes;
    for (const auto& original_shape : original_shapes) {
        std::vector<ov::Shape> one_result_shapes;
        if (single_plane) {
            auto shape = original_shape;
            shape[1] = shape[1] * 3 / 2;
            one_result_shapes.push_back(shape);
        } else {
            auto shape = original_shape;
            one_result_shapes.push_back(shape);
            auto uvShape = ov::Shape{shape[0], shape[1] / 2, shape[2] / 2, 1};
            one_result_shapes.push_back(uvShape);
            one_result_shapes.push_back(uvShape);
        }
        result_shapes.push_back(one_result_shapes);
    }
    return result_shapes;
};

auto in_shapes_single_plane_static = generate_input_static_shapes(inShapes_nhwc, true);
auto in_shapes_three_planes_static = generate_input_static_shapes(inShapes_nhwc, false);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420SinglePlane,
                         ConvertColorI420LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_single_plane_static)),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::Bool(),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvertColorI420LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420ThreePlanes,
                         ConvertColorI420LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_three_planes_static)),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::Bool(),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvertColorI420LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420SinglePlane_acc,
                         ConvertColorI420LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                generate_input_static_shapes({{1, 16 * 6, 16, 1}}, true))),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Bool(),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvertColorI420LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420ThreePlanes_acc,
                         ConvertColorI420LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                generate_input_static_shapes({{1, 16 * 6, 16, 1}}, false))),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Bool(),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvertColorI420LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorI420_acc,
                         ConvertColorI420LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                generate_input_static_shapes({{1, 256 * 256, 256, 1}}, true))),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvertColorI420LayerTest::getTestCaseName);

}  // namespace
