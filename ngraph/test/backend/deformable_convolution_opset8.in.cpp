// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

static void DeformableConvolutionOpset8Test(const std::vector<float>& inputs,
                                            const Shape inputs_shape,
                                            const std::vector<float>& offsets,
                                            const Shape offsets_shape,
                                            const std::vector<float>& filter,
                                            const Shape filter_shape,
                                            const std::vector<float>& outputs,
                                            const Shape outputs_shape,
                                            const Strides& strides,
                                            const CoordinateDiff& padding,
                                            const Strides& dilations,
                                            const int64_t group = 1,
                                            const int64_t deformable_group = 1,
                                            const size_t tolerance_bits = 2,
                                            const bool use_bilinear_interpolation_padding = false)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};
    auto inputs_param = make_shared<op::Parameter>(element::f32, inputs_shape);
    auto offsets_param = make_shared<op::Parameter>(element::f32, offsets_shape);
    auto filter_param = make_shared<op::Parameter>(element::f32, filter_shape);
    auto conv = make_shared<op::v8::DeformableConvolution>(inputs_param,
                                                           offsets_param,
                                                           filter_param,
                                                           strides,
                                                           pads_begin,
                                                           pads_end,
                                                           dilations,
                                                           auto_pad,
                                                           group,
                                                           deformable_group,
                                                           use_bilinear_interpolation_padding);
    auto f =
            make_shared<Function>(conv, ParameterVector{inputs_param, offsets_param, filter_param});
    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs);
    test_case.add_input<float>(offsets);
    test_case.add_input<float>(filter);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run(tolerance_bits);
}

static void DeformableConvolutionOpset8Test(const std::vector<float>& inputs,
                                      const Shape inputs_shape,
                                      const std::vector<float>& offsets,
                                      const Shape offsets_shape,
                                      const std::vector<float>& filter,
                                      const Shape filter_shape,
                                      const std::vector<float>& mask,
                                      const Shape mask_shape,
                                      const std::vector<float>& outputs,
                                      const Shape outputs_shape,
                                      const Strides& strides,
                                      const CoordinateDiff& padding,
                                      const Strides& dilations,
                                      const int64_t group = 1,
                                      const int64_t deformable_group = 1,
                                      const size_t tolerance_bits = 2,
                                      const bool use_bilinear_interpolation_padding = false)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};
    auto inputs_param = make_shared<op::Parameter>(element::f32, inputs_shape);
    auto offsets_param = make_shared<op::Parameter>(element::f32, offsets_shape);
    auto filter_param = make_shared<op::Parameter>(element::f32, filter_shape);
    auto mask_param = make_shared<op::Parameter>(element::f32, mask_shape);
    auto conv = make_shared<op::v8::DeformableConvolution>(inputs_param,
                                                           offsets_param,
                                                           filter_param,
                                                           mask_param,
                                                           strides,
                                                           pads_begin,
                                                           pads_end,
                                                           dilations,
                                                           auto_pad,
                                                           group,
                                                           deformable_group,
                                                           use_bilinear_interpolation_padding);
    auto f =
            make_shared<Function>(conv, ParameterVector{inputs_param, offsets_param, filter_param, mask_param});
    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs);
    test_case.add_input<float>(offsets);
    test_case.add_input<float>(filter);
    test_case.add_input<float>(mask);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run(tolerance_bits);
}
// clang-format off

// regular convolution attributes (zeroed offsets)
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_default)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    -1.0f, -2.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{-12.0f, -12.0f, -12.0f,
                                     -12.0f, -12.0f, -12.0f,
                                     -12.0f, -12.0f, -12.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 3, 3};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f,
                                    7.0f, 5.0f, 3.0f,
                                    1.0f, 3.0f, 5.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    0.0f, 1.0f};

    const Shape offsets_shape{1, 8, 4, 4};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{1.0f, 3.0f, 5.0f, 0.0f,
                                     9.0f, 12.0f, 16.0f, 5.0f,
                                     15.0f, 20.0f, 16.0f, 3.0f,
                                     2.0f, 7.0f, 13.0f, 5.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_stride)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 5, 5};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 1.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{57.0f, 94.0f,
                                     66.0f, 102.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_dilation)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{2, 2};

    const Shape inputs_shape{1, 1, 7, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f};

    const Shape offsets_shape{1, 18, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{78.0f, 106.0f, 134.0f,
                                     44.0f, 16.0f, -12.0f,
                                     80.0f, 84.0f, 88.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_padding_strides_dilation)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{2, 2};
    const Strides dilations{2, 2};

    const Shape inputs_shape{1, 1, 7, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f};

    const Shape offsets_shape{1, 18, 4, 4};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{15.0f, 38.0f, 70.0f, 66.0f,
                                     33.0f, 78.0f, 134.0f, 103.0f,
                                     40.0f, 80.0f, 88.0f, 58.0f,
                                     30.0f, 56.0f, 72.0f, 34.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_input_channels)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 2, 4, 4};
    const std::vector<float> inputs{
            // channel 1
            1.0f, 3.0f, 5.0f, 7.0f,
            7.0f, 5.0f, 3.0f, 1.0f,
            2.0f, 4.0f, 6.0f, 8.0f,
            8.0f, 6.0f, 4.0f, 2.0f,
            // channel 2
            -1.0f, 3.0f, -5.0f, 7.0f,
            7.0f, -5.0f, 3.0f, -1.0f,
            -2.0f, 4.0f, -6.0f, 8.0f,
            8.0f, -6.0f, 4.0f, -2.0f};

    const Shape filter_shape{1, 2, 3, 3};
    const std::vector<float> filter{
            // channel 1
            5.0f, 3.0f, 5.0f,
            1.0f, 3.0f, 1.0f,
            4.0f, 2.0f, 4.0f,
            // channel 2
            -5.0f, 3.0f, 5.0f,
            1.0f, -3.0f, 1.0f,
            4.0f, 2.0f, -4.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{142.0f, 102.0f,
                                     94.0f, 160.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_output_channels)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{
            1.0f, 3.0f, 5.0f, 7.0f,
            7.0f, 5.0f, 3.0f, 1.0f,
            2.0f, 4.0f, 6.0f, 8.0f,
            8.0f, 6.0f, 4.0f, 2.0f};

    const Shape filter_shape{2, 1, 3, 3};
    const std::vector<float> filter{
            // channel 1
            5.0f, 3.0f, 5.0f,
            1.0f, 3.0f, 1.0f,
            4.0f, 2.0f, 4.0f,
            // channel 2
            -5.0f, 3.0f, 5.0f,
            1.0f, -3.0f, 1.0f,
            4.0f, 2.0f, -4.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{
            // channel 1
            104.0f, 140.0f,
            145.0f, 109.0f,
            // channel 2
            16.0f, 28.0f,
            19.0f, 7.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_batch)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{2, 1, 4, 4};
    const std::vector<float> inputs{
            // batch 1
            1.0f, 3.0f, 2.0f, 1.0f,
            1.0f, 3.0f, 3.0f, 1.0f,
            2.0f, 1.0f, 1.0f, 3.0f,
            3.0f, 2.0f, 3.0f, 3.0f,
            // batch 2
            -1.0f, 3.0f, 2.0f, -1.0f,
            1.0f, 3.0f, -3.0f, 1.0f,
            -2.0f, -1.0f, 1.0f, 3.0f,
            3.0f, 2.0f, 3.0f, -3.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{-5.0f, 3.0f, 5.0f,
                                    1.0f, -3.0f, 1.0f,
                                    4.0f, 2.0f, -4.0f};

    const Shape offsets_shape{2, 18, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{2, 1, 2, 2};
    const std::vector<float> outputs{
            // batch 1
            15.0f, -15.0f,
            23.0f, 2.0f,
            // batch 2
            -1.0f, -15.0f,
            -5.0f, 6.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

// group & deformable_group attributes (zeroed offsets)
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_groups_basic)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{ // channel 1
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            // channel 2
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,
            // channel 3
            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            // channel 4
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{ // filter 1 channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // filter 1 channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // filter 2 channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // filter 2 channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 8, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{ // channel 1
            356.0f, 392.0f,
            464.0f, 500.0f,
            // channel 2
            -1004.0f, -1040.0f,
            -1112.0f, -1148.0f};
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding, dilations, group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_groups_complex)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 4;

    const Shape inputs_shape{1, 8, 3, 3};
    const std::vector<float> inputs{ // channel 1
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            // channel 2
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,
            // channel 3
            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            // channel 4
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f,
            // channel 5
            37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f,
            43.0f, 44.0f, 45.0f,
            // channel 6
            46.0f, 47.0f, 48.0f,
            49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f,
            // channel 7
            55.0f, 56.0f, 57.0f,
            58.0f, 59.0f, 60.0f,
            61.0f, 62.0f, 63.0f,
            // channel 8
            64.0f, 65.0f, 66.0f,
            67.0f, 68.0f, 69.0f,
            70.0f, 71.0f, 72.0f,};

    const Shape filter_shape{4, 2, 2, 2};
    const std::vector<float> filter{ // filter 1 channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // filter 1 channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // filter 2 channel 1
            9.0f, 10.0f,
            11.0f, 12.0f,
            // filter 2 channel 2
            13.0f, 14.0f,
            15.0f, 16.0f,
            // filter 3 channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // filter 3 channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // filter 4 channel 1
            -9.0f, -10.0f,
            -11.0f, -12.0f,
            // filter 4 channel 2
            -13.0f, -14.0f,
            -15.0f, -16.0f};

    const Shape offsets_shape{1, 8, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 4, 2, 2};
    const std::vector<float> outputs{ // channel 1
            356.0f, 392.0f,
            464.0f, 500.0f,
            // channel 2
            2636.0f, 2736.0f,
            2936.0f, 3036.0f,
            // channel 3
            -1652.0f, -1688.0f,
            -1760.0f, -1796.0f,
            // channel 4
            -6236.0f, -6336.0f,
            -6536.0f, -6636.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding, dilations, group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_deforgroup)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 2, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{1, 2, 2, 2};
    const std::vector<float> filter{// channel 1
            1.0f, 2.0f,
            -1.0f, -2.0f,
            // channel 2
            3.0f, 4.0f,
            -3.0f, -4.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{-40.0f, -40.0f, -40.0f,
                                     -40.0f, -40.0f, -40.0f,
                                     -40.0f, -40.0f, -40.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_groups_and_deforgroups)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f,
                                    22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f,
                                    28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    3.0f, 4.0f,
                                    5.0f, 6.0f,
                                    7.0f, 8.0f,
                                    -1.0f, -2.0f,
                                    -3.0f, -4.0f,
                                    -5.0f, -6.0f,
                                    -7.0f, -8.0f,
    };

    const Shape offsets_shape{1, 16, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{356.0f, 392.0f,
                                     464.0f, 500.0f,
                                     -1004.0f, -1040.0f,
                                     -1112.0f, -1148.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding,
                              dilations, group, deformable_group);
}

// deformable convolution atrributes (integral offsets)
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_default)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    -1.0f, -2.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets{// window 1 (Y=0, X=0) -> Y coordinate
            1.0f, 1.0f, 1.0f, // out1 .. out 3
            1.0f, 1.0f, 1.0f, // out4 .. out 6
            1.0f, 1.0f, 1.0f, // out7 .. out 9
            // window 1 (Y=0, X=0) -> X coordinate
            1.0f, 1.0f, 1.0f, // out1 .. out 3
            1.0f, 1.0f, 1.0f, // out4 .. out 6
            1.0f, 1.0f, 1.0f, // out7 .. out 9
            // window 2
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            // window 2
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            // window 3
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            // window 3
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            // window 4
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            // window 4
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
    };

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{-12.0f, -12.0f, -4.0f,
                                     -12.0f, -12.0f, -4.0f,
                                     44.0f, 47.0f, 16.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 7.0f, 7.0f,
                                    7.0f, 6.0f, 3.0f, 1.0f,
                                    4.0f, 4.0f, 2.0f, 8.0f,
                                    1.0f, 1.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape offsets_shape{1, 18, 4, 4};
    const std::vector<float> offsets{1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{56.0f, 39.0f, 44.0f, 18.0f,
                                     38.0f, 56.0f, 65.0f, 0.0f,
                                     19.0f, 38.0f, 20.0f, 20.0f,
                                     6.0f, 19.0f, 33.0f, 0.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_stride)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 5, 5};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 1.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets{0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f,
                                     0.0f, 2.0f,
                                     1.0f, 0.0f};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{57.0f, 40.0f,
                                     38.0f, 102.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_dilation)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{2, 2};

    const Shape inputs_shape{1, 1, 7, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f};

    const Shape offsets_shape{1, 18, 3, 3};
    const std::vector<float> offsets{1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f,
                                     1.0f, 2.0f, 0.0f,
                                     0.0f, 2.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{16.0f, -2.0f, 134.0f,
                                     44.0f, -4.0f, -12.0f,
                                     10.0f, 84.0f, -4.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_padding_stride_dilation)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{2, 2};
    const Strides dilations{2, 2};

    const Shape inputs_shape{1, 1, 7, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f};

    const Shape offsets_shape{1, 18, 4, 4};
    const std::vector<float> offsets{1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,

                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{15.0f, 38.0f, 2.0f, 66.0f,
                                     26.0f, 78.0f, 134.0f, 16.0f,
                                     23.0f, 80.0f, -4.0f, 58.0f,
                                     13.0f, 56.0f, 72.0f, -4.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_input_channels)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 2, 4, 4};
    const std::vector<float> inputs{
            // channel 1
            1.0f, 3.0f, 5.0f, 7.0f,
            7.0f, 5.0f, 3.0f, 1.0f,
            2.0f, 4.0f, 6.0f, 8.0f,
            8.0f, 6.0f, 4.0f, 2.0f,
            // channel 2
            -1.0f, 3.0f, -5.0f, 7.0f,
            7.0f, -5.0f, 3.0f, -1.0f,
            -2.0f, 4.0f, -6.0f, 8.0f,
            8.0f, -6.0f, 4.0f, -2.0f};

    const Shape filter_shape{1, 2, 3, 3};
    const std::vector<float> filter{
            // channel 1
            5.0f, 3.0f, 5.0f,
            1.0f, 3.0f, 1.0f,
            4.0f, 2.0f, 4.0f,
            // channel 2
            -5.0f, 3.0f, 5.0f,
            1.0f, -3.0f, 1.0f,
            4.0f, 2.0f, -4.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets{1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f,
                                     1.0f, 1.0f,
                                     0.0f, 2.0f};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{160.0f, 32.0f,
                                     94.0f, 20.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_output_channels)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f};

    const Shape filter_shape{2, 1, 2, 2};
    const std::vector<float> filter{ // filter 1
            1.0f, 2.0f,
            -1.0f, -2.0f,
            // filter 2
            3.0f, 4.0f,
            -3.0f, -4.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets{//channel 1: Y offsets
            1.0f, 1.0f, 1.0f, // out1 .. out 3
            1.0f, 1.0f, 1.0f, // out4 .. out 6
            1.0f, 1.0f, 1.0f, // out7 .. out 9
            //channel 1: X offsets
            1.0f, 1.0f, 1.0f, // out1 .. out 3
            1.0f, 1.0f, 1.0f, // out4 .. out 6
            1.0f, 1.0f, 1.0f, // out7 .. out 9
            //channel 2: Y offsets
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //channel 2: X offsets
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //channel 3: Y offsets
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //channel 3: X offsets
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //channel 4: Y offsets
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //channel 4: X offsets
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
    };

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{
            // output 1
            -12.0f, -12.0f, -4.0f,
            -12.0f, -12.0f, -4.0f,
            44.0f, 47.0f, 16.0f,
            // output 2
            -28.0f, -28.0f, -12.0f,
            -28.0f, -28.0f, -12.0f,
            102.0f, 109.0f, 48.0f, };

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_batch)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{2, 1, 4, 4};
    const std::vector<float> inputs{//batch 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            //batch 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    -1.0f, -2.0f};

    const Shape offsets_shape{2, 8, 3, 3};
    const std::vector<float> offsets{// batch1
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            // batch2
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
    };

    const Shape outputs_shape{2, 1, 3, 3};
    const std::vector<float> outputs{// batch 1
            -12.0f, -12.0f, -4.0f,
            -12.0f, -12.0f, -4.0f,
            44.0f, 47.0f, 16.0f,
            // batch 2
            -12.0f, -12.0f, -12.0f,
            -12.0f, -12.0f, -12.0f,
            -12.0f, -12.0f, -12.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}
// group & deformable_group attributes (integral offsets)
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_groups_basic)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{ // channel 1
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            // channel 2
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,
            // channel 3
            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            // channel 4
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{ // filter 1 channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // filter 1 channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // filter 2 channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // filter 2 channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 8, 2, 2};
    const std::vector<float> offsets {
            // window 1 (F_Y=0, F_X=0) -> I_Y coordinate
            1.0f, 0.0f, 2.0f, 1.0f, // out1 .. out 4
            // window 1 (F_Y=0, F_X=0) -> I_X coordinate
            0.0f, 1.0f, 1.0f, 2.0f, // out1 .. out 4
            // window 2 (F_Y=0, F_X=1) -> I_Y coordinate
            1.0f, 1.0f, 1.0f, 1.0f, // out1 .. out 4
            // window 2 (F_Y=0, F_X=1) -> I_X coordinate
            1.0f, 1.0f, 1.0f, 1.0f, // out1 .. out 4
            // window 3 (F_Y=1, F_X=0) -> I_Y coordinate
            2.0f, 2.0f, 2.0f, 2.0f, // out1 .. out 4
            // window 3 (F_Y=1, F_X=0) -> I_X coordinate
            2.0f, 2.0f, 2.0f, 2.0f, // out1 .. out 4
            // window 4 (F_Y=1, F_X=1) -> I_Y coordinate
            2.0f, 2.0f, 2.0f, 2.0f, // out1 .. out 4
            // window 4 (F_Y=1, F_X=1) -> I_X coordinate
            2.0f, 2.0f, 2.0f, 2.0f}; // out1 .. out 4

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{ // channel 1
            171.0f, 63.0f,
            126.0f, 0.0f,
            // channel 2
            -423.0f, -171.0f,
            -270.0f, 0.0f};
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding, dilations, group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_groups_complex)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 4;

    const Shape inputs_shape{1, 8, 3, 3};
    const std::vector<float> inputs{ // channel 1
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            // channel 2
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,
            // channel 3
            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            // channel 4
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f,
            // channel 5
            37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f,
            43.0f, 44.0f, 45.0f,
            // channel 6
            46.0f, 47.0f, 48.0f,
            49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f,
            // channel 7
            55.0f, 56.0f, 57.0f,
            58.0f, 59.0f, 60.0f,
            61.0f, 62.0f, 63.0f,
            // channel 8
            64.0f, 65.0f, 66.0f,
            67.0f, 68.0f, 69.0f,
            70.0f, 71.0f, 72.0f,};

    const Shape filter_shape{4, 2, 2, 2};
    const std::vector<float> filter{ // filter 1 channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // filter 1 channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // filter 2 channel 1
            9.0f, 10.0f,
            11.0f, 12.0f,
            // filter 2 channel 2
            13.0f, 14.0f,
            15.0f, 16.0f,
            // filter 3 channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // filter 3 channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // filter 4 channel 1
            -9.0f, -10.0f,
            -11.0f, -12.0f,
            // filter 4 channel 2
            -13.0f, -14.0f,
            -15.0f, -16.0f};

    const Shape offsets_shape{1, 8, 2, 2};
    const std::vector<float> offsets {1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f};

    const Shape outputs_shape{1, 4, 2, 2};
    const std::vector<float> outputs{ // channel 1
            500.0f, 234.0f,
            219.0f, 99.0f,
            // channel 2
            3036.0f, 1482.0f,
            1463.0f, 711.0f,
            // channel 3
            -1796.0f, -810.0f,
            -723.0f, -315.0f,
            // channel 4
            -6636.0f, -3210.0f,
            -3119.0f, -1503.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding, dilations, group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_deforgroup_basic)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 2, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{// f1: channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f2: channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 16, 3, 3};
    const std::vector<float> offsets{// defgroup 1
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //  defgroup 2
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
    };

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{// output 1
            610.0f, 646.0f, 612.0f,
            754.0f, 790.0f, 732.0f,
            768.0f, 797.0f, 792.0f,
            // output 2
            -610.0f, -646.0f, -612.0f,
            -754.0f, -790.0f, -732.0f,
            -768.0f, -797.0f, -792.0f,
    };

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_deforgroup_complex1)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 4;

    const Shape inputs_shape{1, 4, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f,
            // channel 3
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 4
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{2, 4, 2, 2};
    const std::vector<float> filter{// f1: channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f1: channel 3
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 4
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f2: channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // f2: channel 3
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 4
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 32, 3, 3};
    const std::vector<float> offsets{// defgroup 1
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //  defgroup 2
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            // defgroup 3
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //  defgroup 4
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
    };

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{// output 1
            1220.0f, 1292.0f, 1224.0f,
            1508.0f, 1580.0f, 1464.0f,
            1536.0f, 1594.0f, 1584.0f,
            // output 2
            -1220.0f, -1292.0f, -1224.0f,
            -1508.0f, -1580.0f, -1464.0f,
            -1536.0f, -1594.0f, -1584.0f,
    };

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_deforgroup_complex2)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 4, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f,
            // channel 3
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 4
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{2, 4, 2, 2};
    const std::vector<float> filter{// f1: channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f1: channel 3
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 4
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f2: channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // f2: channel 3
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 4
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 16, 3, 3};
    const std::vector<float> offsets{// defgroup 1
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //  defgroup 2
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
    };

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{// output 1
            1300.0f, 1372.0f, 992.0f,
            1588.0f, 1660.0f, 1200.0f,
            1228.0f, 1278.0f, 1096.0f,
            // output 2
            -1300.0f, -1372.0f, -992.0f,
            -1588.0f, -1660.0f, -1200.0f,
            -1228.0f, -1278.0f, -1096.0f,
    };

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_integral_offsets_groups_and_deforgroups)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f,
                                    22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f,
                                    28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    3.0f, 4.0f,
                                    5.0f, 6.0f,
                                    7.0f, 8.0f,
                                    -1.0f, -2.0f,
                                    -3.0f, -4.0f,
                                    -5.0f, -6.0f,
                                    -7.0f, -8.0f,
    };

    const Shape offsets_shape{1, 16, 2, 2};
    const std::vector<float> offsets{// defgroup 1
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            // defgroup 2
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
    };

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{500.0f, 234.0f,
                                     219.0f, 99.0f,
                                     -1004.0f, -1040.0f,
                                     -1112.0f, -1148.0f};

    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding,
                              dilations, group, deformable_group);
}

// deformable convolution atrributes (real offsets)
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_default)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    -1.0f, -2.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{-11.999998f, -11.999999f, -4.000000f,
                                     -10.799999f, -10.800001f, -3.600004f,
                                     44.300000f, 47.100000f, 16.000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 7.0f, 7.0f,
                                    7.0f, 6.0f, 3.0f, 1.0f,
                                    4.0f, 4.0f, 2.0f, 8.0f,
                                    1.0f, 1.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape offsets_shape{1, 18, 4, 4};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{54.870006f, 61.630001f, 43.230003f, 28.600002f,
                                     35.590000f, 25.819999f, 20.880001f, 7.700000f,
                                     19.089998f, 31.719999f, 19.250000f, 7.399999f,
                                     6.299999f, 9.199999f, 5.099999f, 2.000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_stride)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 1, 5, 5};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 1.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{61.229999f, 29.509998f,
                                     39.640003f, 22.640003f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_padding_stride_dilation)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{2, 2};
    const Strides dilations{2, 2};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 1, 7, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filter{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f};

    const Shape offsets_shape{1, 18, 4, 4};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{15.260000f, 24.119997f, 6.439994f, -3.940005f,
                                     26.440002f, 20.319999f, -0.500001f, -11.720002f,
                                     23.500003f, 14.040000f, -1.279998f, -3.860000f,
                                     12.500000f, -2.599999f, -5.299999f, -3.099999f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_input_channels)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 2, 4, 4};
    const std::vector<float> inputs{
            // channel 1
            1.0f, 3.0f, 5.0f, 7.0f,
            7.0f, 5.0f, 3.0f, 1.0f,
            2.0f, 4.0f, 6.0f, 8.0f,
            8.0f, 6.0f, 4.0f, 2.0f,
            // channel 2
            -1.0f, 3.0f, -5.0f, 7.0f,
            7.0f, -5.0f, 3.0f, -1.0f,
            -2.0f, 4.0f, -6.0f, 8.0f,
            8.0f, -6.0f, 4.0f, -2.0f};

    const Shape filter_shape{1, 2, 3, 3};
    const std::vector<float> filter{
            // channel 1
            5.0f, 3.0f, 5.0f,
            1.0f, 3.0f, 1.0f,
            4.0f, 2.0f, 4.0f,
            // channel 2
            -5.0f, 3.0f, 5.0f,
            1.0f, -3.0f, 1.0f,
            4.0f, 2.0f, -4.0f};

    const Shape offsets_shape{1, 18, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{148.000000f, 43.259998f,
                                     91.279998f, 111.199996f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_output_channels)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f};

    const Shape filter_shape{2, 1, 2, 2};
    const std::vector<float> filter{ // filter 1
            1.0f, 2.0f,
            -1.0f, -2.0f,
            // filter 2
            3.0f, 4.0f,
            -3.0f, -4.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{
            // output 1
            -12.000000f, -12.000000f, -4.000000f,
            -10.799999f, -10.799995f, -3.600000f,
            44.299999f, 47.099998f, 16.000000f,
            // output 2
            -28.000000f, -28.000000f, -12.000000f,
            -25.200000f, -25.199993f, -10.800003f,
            102.699996f, 109.300003f, 48.000000f, };

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_batch)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{2, 1, 4, 4};
    const std::vector<float> inputs{//batch 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            //batch 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    -1.0f, -2.0f};

    const Shape offsets_shape{2, 8, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{2, 1, 3, 3};
    const std::vector<float> outputs{// batch 1
            -12.000000f, -12.000000f, -4.000000f,
            -10.799999f, -10.799995f, -3.600000f,
            44.299999f, 47.099998f, 16.000000f,
            // batch 2
            -12.000000f, -12.000000f, -4.000000f,
            -10.799999f, -10.799995f, -3.600000f,
            92.300003f, 95.099998f, 32.000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

// group & deformable_group attributes (real offsets)
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_groups_basic)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{ // channel 1
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            // channel 2
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,
            // channel 3
            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            // channel 4
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{ // filter 1 channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // filter 1 channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // filter 2 channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // filter 2 channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 8, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{ // channel 1
            505.800020f, 235.800000f,
            219.600000f, 99.000000f,
            // channel 2
            -1153.800000f, -523.800000f,
            -471.600000f, -207.0000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_groups_complex)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 4;
    const int64_t deformable_group = 1;

    const Shape inputs_shape{1, 8, 3, 3};
    const std::vector<float> inputs{ // channel 1
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            // channel 2
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,
            // channel 3
            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            // channel 4
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f,
            // channel 5
            37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f,
            43.0f, 44.0f, 45.0f,
            // channel 6
            46.0f, 47.0f, 48.0f,
            49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f,
            // channel 7
            55.0f, 56.0f, 57.0f,
            58.0f, 59.0f, 60.0f,
            61.0f, 62.0f, 63.0f,
            // channel 8
            64.0f, 65.0f, 66.0f,
            67.0f, 68.0f, 69.0f,
            70.0f, 71.0f, 72.0f,};

    const Shape filter_shape{4, 2, 2, 2};
    const std::vector<float> filter{ // filter 1 channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // filter 1 channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // filter 2 channel 1
            9.0f, 10.0f,
            11.0f, 12.0f,
            // filter 2 channel 2
            13.0f, 14.0f,
            15.0f, 16.0f,
            // filter 3 channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // filter 3 channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // filter 4 channel 1
            -9.0f, -10.0f,
            -11.0f, -12.0f,
            // filter 4 channel 2
            -13.0f, -14.0f,
            -15.0f, -16.0f};

    const Shape offsets_shape{1, 8, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 4, 2, 2};
    const std::vector<float> outputs{ // channel 1
            505.800020f, 235.800000f,
            219.600000f, 99.000000f,
            // channel 2
            3054.600000f, 1488.600000f,
            1465.200100f, 711.000000f,
            // channel 3
            -1801.799900f, -811.80000f,
            -723.600000f, -315.000000f,
            // channel 4
            -6654.600000f, -3216.600000f,
            -3121.200000f, -1503.000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_deforgroup_basic)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 2, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{// f1: channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f2: channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 16, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{// output 1
            758.000000f, 792.000000f, 366.399993f,
            893.200012f, 927.200012f, 426.399993f,
            381.399993f, 394.600006f, 176.000000f,
            // output 2
            -758.000000f, -792.000000f, -366.399993f,
            -893.200012f, -927.200012f, -426.399993f,
            -381.399993f, -394.600006f, -176.000000f,
    };

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_deforgroup_complex1)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 4;

    const Shape inputs_shape{1, 4, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f,
            // channel 3
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 4
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{2, 4, 2, 2};
    const std::vector<float> filter{// f1: channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f1: channel 3
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 4
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f2: channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // f2: channel 3
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 4
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 32, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{// output 1
            1516.000000f, 1583.999877f, 732.799987f,
            1786.400146f, 1854.400024f, 852.799987f,
            762.799987f, 789.200012f, 352.000000f,
            // output 2
            -1516.000000f, -1583.999877f, -732.799987f,
            -1786.400146f, -1854.400024f, -852.799987f,
            -762.799987f, -789.200012f, -352.000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_deforgroup_complex2)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 1;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 4, 4, 4};
    const std::vector<float> inputs{// channel 1
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 2
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f,
            // channel 3
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            // channel 4
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,
            29.0f, 30.0f, 31.0f, 32.0f};

    const Shape filter_shape{2, 4, 2, 2};
    const std::vector<float> filter{// f1: channel 1
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 2
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f1: channel 3
            1.0f, 2.0f,
            3.0f, 4.0f,
            // f1: channel 4
            5.0f, 6.0f,
            7.0f, 8.0f,
            // f2: channel 1
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 2
            -5.0f, -6.0f,
            -7.0f, -8.0f,
            // f2: channel 3
            -1.0f, -2.0f,
            -3.0f, -4.0f,
            // f2: channel 4
            -5.0f, -6.0f,
            -7.0f, -8.0f};

    const Shape offsets_shape{1, 16, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{// output 1
            1516.000000f, 1583.999877f, 732.799987f,
            1786.400146f, 1854.400024f, 852.799987f,
            762.799987f, 789.200012f, 352.000000f,
            // output 2
            -1516.000000f, -1583.999877f, -732.799987f,
            -1786.400146f, -1854.400024f, -852.799987f,
            -762.799987f, -789.200012f, -352.000000f,
    };

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape,strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_real_offsets_groups_and_deforgroups)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f,
                                    22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f,
                                    28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    3.0f, 4.0f,
                                    5.0f, 6.0f,
                                    7.0f, 8.0f,
                                    -1.0f, -2.0f,
                                    -3.0f, -4.0f,
                                    -5.0f, -6.0f,
                                    -7.0f, -8.0f,
    };

    const Shape offsets_shape{1, 16, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1f);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{505.800020f, 235.800000f,
                                     219.600000f, 99.000000f,
                                     -1153.800000f, -523.800000f,
                                     -471.600000f, -207.000000f};

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                              filter_shape, outputs, outputs_shape, strides, padding,
                              dilations, group, deformable_group, tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_v8_zeroed_offsets_default_modulation_scalars)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    -1.0f, -2.0f};

    const Shape offsets_shape{1, 8, 3, 3};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 0);

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{-6.0f, -6.0f, -6.0f,
                                     -6.0f, -6.0f, -6.0f,
                                     -6.0f, -6.0f, -6.0f};

    const Shape mask_shape{1, 4, 3, 3};
    const std::vector<float> mask{0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f,
                                  0.5f, 0.5f, 0.5f};
    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                                filter_shape, mask, mask_shape, outputs, outputs_shape,strides, padding, dilations,
                                1, 1, tolerance_bits, true);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_opset8_2D_zeroed_offsets_groups_and_deforgroups_modulation_scalars)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    const Shape inputs_shape{1, 4, 3, 3};
    const std::vector<float> inputs{1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f,
                                    22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f,
                                    28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filter{1.0f, 2.0f,
                                    3.0f, 4.0f,
                                    5.0f, 6.0f,
                                    7.0f, 8.0f,
                                    -1.0f, -2.0f,
                                    -3.0f, -4.0f,
                                    -5.0f, -6.0f,
                                    -7.0f, -8.0f,
    };

    const Shape offsets_shape{1, 16, 2, 2};
    const std::vector<float> offsets(ngraph::shape_size(offsets_shape), 1.1);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{220.15443f ,   38.199608f,
                                     32.643005f,   59.340614f,
                                     -419.0005f  , -252.08015f,
                                     -182.44444f , -165.99335f};

    const Shape mask_shape{1, 8, 2, 2};
    const std::vector<float> mask{0.64f,
                                  0.18f,
                                  0.23f,
                                  0.74f,
                                  0.89f,
                                  0.70f,
                                  0.13f,
                                  0.99f,
                                  0.48f,
                                  0.20f,
                                  0.67f,
                                  0.88f,
                                  0.17f,
                                  0.19f,
                                  0.53f,
                                  0.22f,
                                  0.50f,
                                  0.07f,
                                  0.21f,
                                  0.99f,
                                  0.09f,
                                  0.28f,
                                  0.66f,
                                  0.91f,
                                  0.28f,
                                  0.89f,
                                  0.91f,
                                  0.39f,
                                  0.70f,
                                  0.67f,
                                  0.26f,
                                  0.09f
    };

    const size_t tolerance_bits = 6;
    DeformableConvolutionOpset8Test(inputs, inputs_shape, offsets, offsets_shape, filter,
                                filter_shape, mask, mask_shape, outputs, outputs_shape, strides, padding,
                                dilations, group, deformable_group, tolerance_bits, true);
}

