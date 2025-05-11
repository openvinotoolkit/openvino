// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/group_convolution_backprop_data.hpp"

namespace {
using ov::test::GroupConvBackpropLayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32};

const std::vector<size_t> num_out_channels = {16, 32};
const std::vector<size_t> num_groups = {2, 8, 16};
const std::vector<ov::Shape> empty_output_shape = {{}};
const std::vector<std::vector<ptrdiff_t >> empty_output_padding = {{}};

/* ============= _1d GroupConvolution ============= */
const std::vector<ov::Shape> input_shapes_1d = {{1, 16, 32}};

const std::vector<std::vector<size_t >> kernels_1d = {{1}, {3}};
const std::vector<std::vector<size_t>> strides_1d = {{1}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_1d = {{0}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_1d = {{0}};
const std::vector<std::vector<size_t>> dilations_1d = {{1}};

const auto groupConvBackpropData_1dParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels_1d),
        ::testing::ValuesIn(strides_1d),
        ::testing::ValuesIn(pad_begins_1d),
        ::testing::ValuesIn(pad_ends_1d),
        ::testing::ValuesIn(dilations_1d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(empty_output_padding)
);

const auto groupConvBackpropData_1dParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels_1d),
        ::testing::ValuesIn(strides_1d),
        ::testing::ValuesIn(pad_begins_1d),
        ::testing::ValuesIn(pad_ends_1d),
        ::testing::ValuesIn(dilations_1d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(empty_output_padding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_1d_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_1dParams_ExplicitPadding,
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_1d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_1d_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_1dParams_AutoPadValid,
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_1d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

/* ============= _2d GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> input_shapes_2d = {{{1, 16, 10, 10}},
                                                             {{1, 32, 10, 10}}};
const std::vector<std::vector<size_t >> kernels_2d = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t >> strides_2d = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_2d = {{0, 0}};
const std::vector<std::vector<size_t >> dilations_2d = {{1, 1}};

const auto groupConvBackpropData_2dParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels_2d),
        ::testing::ValuesIn(strides_2d),
        ::testing::ValuesIn(pad_begins_2d),
        ::testing::ValuesIn(pad_ends_2d),
        ::testing::ValuesIn(dilations_2d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(empty_output_padding)
);
const auto groupConvBackpropData_2dParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels_2d),
        ::testing::ValuesIn(strides_2d),
        ::testing::ValuesIn(pad_begins_2d),
        ::testing::ValuesIn(pad_ends_2d),
        ::testing::ValuesIn(dilations_2d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(empty_output_padding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_2d_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_2dParams_ExplicitPadding,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_2d_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_2dParams_AutoPadValid,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<ov::Shape> input_shape_2d = {{1, 16, 9, 12}};
const std::vector<ov::Shape> output_shapes_2d = {{6, 6}, {4, 9}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_2d_OutputShapeDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_2dParams_AutoPadValid,
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(input_shape_2d)),
                                ::testing::ValuesIn(output_shapes_2d),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<std::vector<ptrdiff_t>> output_padding_2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t >> test_strides_2d = {{3, 3}};

const auto conv_2dParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels_2d),
        ::testing::ValuesIn(test_strides_2d),
        ::testing::ValuesIn(pad_begins_2d),
        ::testing::ValuesIn(pad_ends_2d),
        ::testing::ValuesIn(dilations_2d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(output_padding_2d)
);
const auto conv_2dParams_AutoPadValid_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels_2d),
        ::testing::ValuesIn(test_strides_2d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations_2d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(output_padding_2d)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_2d_ExplicitPadding_output_paddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv_2dParams_AutoPadValid_output_padding,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_2d_AutoPadding_output_paddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv_2dParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

/* ============= _3d GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> input_shapes_3d = {{{1, 16, 5, 5, 5}},
                                                             {{1, 32, 5, 5, 5}}};
const std::vector<std::vector<size_t >> kernels_3d = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t >> strides_3d = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_3d = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_3d = {{0, 0, 0}};
const std::vector<std::vector<size_t >> dilations_3d = {{1, 1, 1}};

const auto groupConvBackpropData_3dParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels_3d),
        ::testing::ValuesIn(strides_3d),
        ::testing::ValuesIn(pad_begins_3d),
        ::testing::ValuesIn(pad_ends_3d),
        ::testing::ValuesIn(dilations_3d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(empty_output_padding)
);
const auto groupConvBackpropData_3dParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels_3d),
        ::testing::ValuesIn(strides_3d),
        ::testing::ValuesIn(pad_begins_3d),
        ::testing::ValuesIn(pad_ends_3d),
        ::testing::ValuesIn(dilations_3d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(empty_output_padding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_3d_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_3dParams_ExplicitPadding,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_3d_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_3dParams_AutoPadValid,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<ov::Shape> input_shape_3d = {{1, 16, 10, 10, 10}};
const std::vector<ov::Shape> output_shapes_3d = {{8, 8, 8}, {10, 10, 10}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_3d_OutputShapeDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData_3dParams_AutoPadValid,
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(input_shape_3d)),
                                ::testing::ValuesIn(output_shapes_3d),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<std::vector<ptrdiff_t>> output_padding_3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<size_t >> test_strides_3d = {{3, 3, 3}};

const auto conv_3dParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels_3d),
        ::testing::ValuesIn(test_strides_3d),
        ::testing::ValuesIn(pad_begins_3d),
        ::testing::ValuesIn(pad_ends_3d),
        ::testing::ValuesIn(dilations_3d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(output_padding_3d)
);
const auto conv_3dParams_AutoPadValid_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels_3d),
        ::testing::ValuesIn(test_strides_3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations_3d),
        ::testing::ValuesIn(num_out_channels),
        ::testing::ValuesIn(num_groups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(output_padding_3d)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_3d_ExplicitPadding_output_paddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv_3dParams_AutoPadValid_output_padding,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData_3d_AutoPadding_output_paddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv_3dParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d)),
                                ::testing::ValuesIn(empty_output_shape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

}  // namespace
