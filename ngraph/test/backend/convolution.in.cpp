//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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

static void ConvolutionTest(const std::vector<float>& inputs,
                            const Shape inputs_shape,
                            const std::vector<float>& filters,
                            const Shape filter_shape,
                            const std::vector<float>& outputs,
                            const Shape outputs_shape,
                            const Strides& strides,
                            const CoordinateDiff& padding,
                            const Strides& dilations)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};

    auto inputs_param = make_shared<op::Parameter>(element::f32, inputs_shape);
    auto filters_param = make_shared<op::Parameter>(element::f32, filter_shape);
    auto conv = make_shared<op::v1::Convolution>(
        inputs_param, filters_param, strides, pads_begin, pads_end, dilations, auto_pad);
    auto f = make_shared<Function>(conv, ParameterVector{inputs_param, filters_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs);
    test_case.add_input<float>(filters);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run();
}

// --------------------- 1D convolution ------------------------------------------
// clang-format off
NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_1channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 6};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4};
    const std::vector<float> outputs{5.0f, 6.0f, 7.0f, 2.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_1channel_padding)
{
    const Strides strides{1};
    const CoordinateDiff padding{1};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4};
    const std::vector<float> outputs{3.0f, 5.0f, 6.0f, 6.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_1channel_stride)
{
    const Strides strides{2};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 5};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f, 1.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 2};
    const std::vector<float> outputs{5.0f, 7.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_1channel_dilation)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{2};

    const Shape inputs_shape{1, 1, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 3};
    const std::vector<float> outputs{3.0f, 8.0f, 9.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_1channel_padding_stride_dilation)
{
    const Strides strides{2};
    const CoordinateDiff padding{2};
    const Strides dilations{2};

    const Shape inputs_shape{1, 1, 7};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4};
    const std::vector<float> outputs{3.0f, 3.0f, 9.0f, 2.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_2channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 2, 4};
    const std::vector<float> inputs{
                                    // channel 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    // channel 2
                                    2.0f, 2.0f, 3.0f, 1.0f};

    const Shape filter_shape{1, 2, 3};
    const std::vector<float> filters{
                                    // channel 1
                                    2.0f, 0.0f, 1.0f,
                                    // channel 2
                                    1.0f, 0.0f, 2.0f};

    const Shape outputs_shape{1, 1, 2};
    const std::vector<float> outputs{12.0f, 11.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_1batch_2filter)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 2.0f, 1.0f};

    const Shape filter_shape{2, 1, 3};
    const std::vector<float> filters{
                                    // filter 1
                                    2.0f, 0.0f, 1.0f,
                                    // filter 2
                                    1.0f, 0.0f, 2.0f};

    const Shape outputs_shape{1, 2, 2};
    const std::vector<float> outputs{
                                    // channel 1
                                    4.0f, 7.0f,
                                    // channel 2
                                    5.0f, 5.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_1D_2batch_1channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{2, 1, 4};
    const std::vector<float> inputs{
                                    // batch 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    // batch 2
                                    2.0f, 2.0f, 3.0f, 1.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{2, 1, 2};
    const std::vector<float> outputs{
                                    // batch 1
                                    4.0f, 7.0f,
                                    // batch 2
                                    7.0f, 5.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

// --------------------- 2D convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{47.0f, 69.0f,
                                     70.0f, 48.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_1channel_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f, 7.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     0.0f, 1.0f, 0.0f,
                                     2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{18.0f, 28.0f, 20.0f, 14.0f,
                                     28.0f, 47.0f, 67.0f, 40.0f,
                                     51.0f, 60.0f, 40.0f, 23.0f,
                                     24.0f, 34.0f, 44.0f, 24.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_1channel_stride)
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
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     1.0f, 1.0f, 1.0f,
                                     3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{57.0f, 94.0f,
                                     66.0f, 102.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_1channel_dilation)
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
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     1.0f, 1.0f, 0.0f,
                                     3.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{78.0f, 106.0f, 134.0f,
                                     44.0f, 16.0f, -12.0f,
                                     80.0f, 84.0f, 88.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_1channel_padding_strides_dilation)
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
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     1.0f, 1.0f, 0.0f,
                                     3.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{15.0f, 38.0f, 70.0f, 66.0f,
                                    33.0f, 78.0f, 134.0f, 103.0f,
                                    40.0f, 80.0f, 88.0f, 58.0f,
                                    30.0f, 56.0f, 72.0f, 34.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_2channel)
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
    const std::vector<float> filters{
                                    // channel 1
                                    5.0f, 3.0f, 5.0f,
                                    1.0f, 3.0f, 1.0f,
                                    4.0f, 2.0f, 4.0f,
                                    // channel 2
                                    -5.0f, 3.0f, 5.0f,
                                    1.0f, -3.0f, 1.0f,
                                    4.0f, 2.0f, -4.0f};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{142.0f, 102.0f,
                                     94.0f, 160.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_1batch_2filter)
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
    const std::vector<float> filters{
                                    // channel 1
                                    5.0f, 3.0f, 5.0f,
                                    1.0f, 3.0f, 1.0f,
                                    4.0f, 2.0f, 4.0f,
                                    // channel 2
                                   -5.0f, 3.0f, 5.0f,
                                    1.0f, -3.0f, 1.0f,
                                    4.0f, 2.0f, -4.0f};

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // channel 1
                                     104.0f, 140.0f,
                                     145.0f, 109.0f,
                                     // channel 2
                                     16.0f, 28.0f,
                                     19.0f, 7.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_2D_2batch_1channel)
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
    const std::vector<float> filters{-5.0f, 3.0f, 5.0f,
                                    1.0f, -3.0f, 1.0f,
                                    4.0f, 2.0f, -4.0f};

    const Shape outputs_shape{2, 1, 2, 2};
    const std::vector<float> outputs{
                                    // batch 1
                                    15.0f, -15.0f,
                                    23.0f, 2.0f,
                                    // batch 2
                                    -1.0f, -15.0f,
                                    -5.0f, 6.0f};


    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

// --------------------- 3D convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_1batch_1channel)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> inputs{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f
                                    };

    const Shape filter_shape{1, 1, 3, 3, 3};
    const std::vector<float> filters{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_1batch_1channel_padding)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{1, 1, 1};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> inputs{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f
                                    };

    const Shape filter_shape{1, 1, 3, 3, 3};
    const std::vector<float> filters{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> outputs{
                                     // depth: 1
                                     16.0f, 28.0f, 26.0f, 16.0f,
                                     32.0f, 46.0f, 44.0f, 20.0f,
                                     40.0f, 62.0f, 52.0f, 34.0f,
                                     20.0f, 18.0f, 30.0f, 20.0f,
                                     // depth: 2
                                     24.0f, 42.0f, 39.0f, 24.0f,
                                     48.0f, 69.0f, 66.0f, 30.0f,
                                     60.0f, 93.0f, 78.0f, 51.0f,
                                     30.0f, 27.0f, 45.0f, 30.0f,
                                     // depth: 3
                                     24.0f, 42.0f, 39.0f, 24.0f,
                                     48.0f, 69.0f, 66.0f, 30.0f,
                                     60.0f, 93.0f, 78.0f, 51.0f,
                                     30.0f, 27.0f, 45.0f, 30.0f,
                                     // depth: 4
                                     16.0f, 28.0f, 26.0f, 16.0f,
                                     32.0f, 46.0f, 44.0f, 20.0f,
                                     40.0f, 62.0f, 52.0f, 34.0f,
                                     20.0f, 18.0f, 30.0f, 20.0f,};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_1batch_1channel_stride)
{
    const Strides strides{2, 2, 2};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 1, 5, 5, 5};
    const std::vector<float> inputs{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 5
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    };

    const Shape filter_shape{1, 1, 3, 3, 3};
    const std::vector<float> filters{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // depth: 1
                                     69.0f, 60.0f,
                                     69.0f, 87.0f,
                                     // depth: 2
                                     69.0f, 60.0f,
                                     69.0f, 87.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_1batch_1channel_padding_strides_dilation)
{
    const Strides strides{2, 2, 2};
    const CoordinateDiff padding{2, 2, 2};
    const Strides dilations{2, 2, 2};

    const Shape inputs_shape{1, 1, 7, 7, 7};
    const std::vector<float> inputs{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 5
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 6
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 7
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    };

    const Shape filter_shape{1, 1, 3, 3, 3};
    const std::vector<float> filters{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> outputs{
                                     // depth: 1
                                     10.0f, 18.0f, 20.0f, 16.0f,
                                     38.0f, 40.0f, 54.0f, 30.0f,
                                     38.0f, 42.0f, 52.0f, 30.0f,
                                     36.0f, 30.0f, 30.0f, 20.0f,
                                     // depth: 2
                                     15.0f, 27.0f, 30.0f, 24.0f,
                                     57.0f, 60.0f, 81.0f, 45.0f,
                                     57.0f, 63.0f, 78.0f, 45.0f,
                                     54.0f, 45.0f, 45.0f, 30.0f,
                                     // depth: 3
                                     15.0f, 27.0f, 30.0f, 24.0f,
                                     57.0f, 60.0f, 81.0f, 45.0f,
                                     57.0f, 63.0f, 78.0f, 45.0f,
                                     54.0f, 45.0f, 45.0f, 30.0f,
                                     // depth: 4
                                     10.0f, 18.0f, 20.0f, 16.0f,
                                     38.0f, 40.0f, 54.0f, 30.0f,
                                     38.0f, 42.0f, 52.0f, 30.0f,
                                     36.0f, 30.0f, 30.0f, 20.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_1batch_2channel)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 2, 4, 4, 4};
    const std::vector<float> inputs{
                                    // -- channel 1 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // -- channel 2 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f
                                    };

    const Shape filter_shape{1, 2, 3, 3, 3};
    const std::vector<float> filters{
                                    // -- channel 1 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // -- channel 2 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f
                                    };

    const Shape outputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // depth: 1
                                     138.0f, 132.0f,
                                     186.0f, 156.0f,
                                     // depth: 2
                                     138.0f, 132.0f,
                                     186.0f, 156.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_1batch_2filter)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> inputs{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f};

    const Shape filter_shape{2, 1, 3, 3, 3};
    const std::vector<float> filters{
                                    // -- filter 1 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // -- filter 2 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f
                                    };

    const Shape outputs_shape{1, 2, 2, 2, 2};
    const std::vector<float> outputs{
                                     // -- out 1 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // -- out 2 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_3D_2batch_1channel)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{2, 1, 4, 4, 4};
    const std::vector<float> inputs{
                                    // -- batch 1 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // -- batch 2 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f};

    const Shape filter_shape{1, 1, 3, 3, 3};
    const std::vector<float> filters{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{2, 1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // -- batch 1 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // -- batch 2 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f};

    ConvolutionTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}
// ----------------------  other tests ------------------------------------------
// clang-format on
NGRAPH_TEST(${BACKEND_NAME}, convolution_outlining)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::v1::Convolution>(
        A, B, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
    auto conv2 = make_shared<op::v1::Convolution>(
        conv1, B, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
    auto f = make_shared<Function>(conv2, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_simple)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::v1::Convolution>(
        A, B, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    auto f = make_shared<Function>(conv1, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{3.0f, 3.0f, 3.0f, 3.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{18.0f, 24.0f, 30.0f, 36.0f, 18.0f, 24.0f, 30.0f, 36.0f};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_simple_padding)
{
    Shape shape_a{1, 1, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{1, 1, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 5, 5};
    auto conv1 = make_shared<op::v1::Convolution>(
        A, B, Strides{1, 1}, CoordinateDiff{1, 1}, CoordinateDiff{2, 2}, Strides{1, 1});

    auto f = make_shared<Function>(conv1, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2.0f});
    auto result = backend->create_tensor(element::f32, shape_r);
    // clang-format off
    vector<float> expected_result{0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 2.0f, 4.0f, 0.0f, 0.0f,
                                  0.0f, 6.0f, 8.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // clang-format on
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

// The purpose of this test is to check if we can allow
// data_batch_shape as a node rather than argument
NGRAPH_TEST(${BACKEND_NAME}, dyn_convolution_backprop_data)
{
    Shape shape_filter{6, 3, 3, 3};
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_delta{2, 6, 3, 3};
    auto deltas = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_data_batch_shape{2, 3, 5, 5};
    auto data_batch_shape =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = Strides{1, 1};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{0, 0};
    auto padding_end = CoordinateDiff{0, 0};

    auto conv1 = make_shared<op::v1::ConvolutionBackpropData>(
        deltas, filters, data_batch_shape, strides, padding_begin, padding_end, dilations);

    auto f = make_shared<Function>(conv1, ParameterVector{deltas, filters, data_batch_shape});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    vector<float> filter, delta, expected_result;

    for (int i = 0; i < 6 * 3 * 3 * 3; i++)
        filter.emplace_back(i);

    for (int i = 0; i < 2 * 6 * 3 * 3; i++)
        delta.emplace_back(i);

    for (int i = 0; i < 2 * 3 * 5 * 5; i++)
        expected_result.emplace_back(i);

    vector<int64_t> shapes = {5, 5};

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_delta);
    copy_data(a, delta);
    auto b = backend->create_tensor(element::f32, shape_filter);
    copy_data(b, filter);
    auto c = backend->create_tensor(element::i64, Shape{shapes.size()}); // dynamic data batch shape
    copy_data(c, shapes);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_FALSE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}
