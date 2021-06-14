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

static void ConvolutionBackpropTest(const std::vector<float>& inputs,
                                    const Shape inputs_shape,
                                    const std::vector<float>& filters,
                                    const Shape filter_shape,
                                    const std::vector<float>& outputs,
                                    const Shape outputs_shape,
                                    const Strides& strides,
                                    const CoordinateDiff& padding,
                                    const Strides& dilations,
                                    const CoordinateDiff& output_padding)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};
    const CoordinateDiff out_padding{output_padding};

    auto inputs_param = make_shared<op::Parameter>(element::f32, inputs_shape);
    auto filters_param = make_shared<op::Parameter>(element::f32, filter_shape);
    auto conv = make_shared<op::v1::ConvolutionBackpropData>(inputs_param,
                                                             filters_param,
                                                             strides,
                                                             pads_begin,
                                                             pads_end,
                                                             dilations,
                                                             auto_pad,
                                                             out_padding);
    auto f = make_shared<Function>(conv, ParameterVector{inputs_param, filters_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs);
    test_case.add_input<float>(filters);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run();
}

// --------------------- 1D convolution ------------------------------------------
// clang-format off
NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_1channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{5.0f, 6.0f, 7.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 6};
    const std::vector<float> outputs{10.0f, 12.0f, 19.0f, 10.0f, 7.0f, 2.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_1channel_padding)
{
    const Strides strides{1};
    const CoordinateDiff padding{1};
    const Strides dilations{1};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{5.0f, 6.0f, 7.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4};
    const std::vector<float> outputs{12.0f, 19.0f, 10.0f, 7.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_1channel_stride)
{
    const Strides strides{2};
    const CoordinateDiff padding{0};
    const Strides dilations{1};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 1, 2};
    const std::vector<float> inputs{5.0f, 7.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 5};
    const std::vector<float> outputs{10.0f, 0.0f, 19.0f, 0.0f, 7.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_1channel_output_padding)
{
    const Strides strides{1};
    const CoordinateDiff padding{1};
    const Strides dilations{1};
    const CoordinateDiff output_padding{1};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{5.0f, 6.0f, 7.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 5};
    const std::vector<float> outputs{12.0f, 19.0f, 10.0f, 7.0f, 2.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_1channel_dilation)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{2};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 1, 3};
    const std::vector<float> inputs{8.0f, 5.0f, 1.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 7};
    const std::vector<float> outputs{16.0f, 10.0f, 2.0f, 0.0f, 8.0f, 5.0f, 1.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_1channel_padding_stride_dilation)
{
    const Strides strides{2};
    const CoordinateDiff padding{2};
    const Strides dilations{2};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{3.0f, 9.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 7};
    const std::vector<float> outputs{18.0f, 0.0f, 5.0f, 0.0f, 13.0f, 0.0f, 1.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_2channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 1, 2};
    const std::vector<float> inputs{10.0f, 3.0f};

    const Shape filter_shape{1, 2, 3};
    const std::vector<float> filters{
                                    // channel 1
                                    2.0f, 0.0f, 1.0f,
                                    // channel 2
                                    1.0f, 0.0f, 2.0f};

    const Shape outputs_shape{1, 2, 4};
    const std::vector<float> outputs{
                                    // channel 1
                                    20.0f, 6.0f, 10.0f, 3.0f,
                                    // channel 2
                                    10.0f, 3.0f, 20.0f, 6.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_1batch_2filter)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{1, 2, 2};
    const std::vector<float> inputs{
                                    // channel 1
                                    4.0f, 7.0f,
                                    // channel 2
                                    5.0f, 5.0f};

    const Shape filter_shape{2, 1, 3};
    const std::vector<float> filters{
                                    // filter 1
                                    2.0f, 0.0f, 1.0f,
                                    // filter 2
                                    1.0f, 0.0f, 2.0f};

    const Shape outputs_shape{1, 1, 4};
    const std::vector<float> outputs{13.0f, 19.0f, 14.0f, 17.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_1D_2batch_1channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};
    const CoordinateDiff output_padding{0};

    const Shape inputs_shape{2, 1, 2};
    const std::vector<float> inputs{
                                    // batch 1
                                    1.0f, 3.0f,
                                    // batch 2
                                    2.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{2, 1, 4};
    const std::vector<float> outputs{
                                    // batch 1
                                    2.0f, 6.0f, 1.0f, 3.0f,
                                    // batch 2
                                    4.0f, 4.0f, 2.0f, 2.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

// --------------------- 2D convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 1, 2, 2};
    const std::vector<float> inputs{1.0f, 3.0f,
                                    7.0f, 5.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{1.0f, 5.0f, 9.0f, 9.0f,
                                    7.0f, 20.0f, 34.0f, 15.0f,
                                    3.0f, 18.0f, 12.0f, 3.0f,
                                    21.0f, 29.0f, 17.0f, 5.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_1channel_output_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{1, 1};

    const Shape inputs_shape{1, 1, 2, 2};
    const std::vector<float> inputs{1.0f, 3.0f,
                                    7.0f, 5.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 1.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{23.0f, 35.0f, 18.0f,
                                    23.0f, 19.0f, 8.0f,
                                    29.0f, 17.0f, 5.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_1channel_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

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
    const std::vector<float> outputs{20.0f, 37.0f, 27.0f, 18.0f,
                                     22.0f, 40.0f, 60.0f, 52.0f,
                                     41.0f, 69.0f, 49.0f, 31.0f,
                                     18.0f, 26.0f, 34.0f, 22.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_1channel_stride)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 1, 2, 2};
    const std::vector<float> inputs{2.0f, 5.0f,
                                    4.0f, 3.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     1.0f, 1.0f, 1.0f,
                                     3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 5, 5};
    const std::vector<float> outputs{2.0f, 4.0f, 11.0f, 10.0f, 15.0f,
                                     2.0f, 2.0f, 7.0f, 5.0f, 5.0f,
                                     10.0f, 12.0f, 32.0f, 16.0f, 14.0f,
                                     4.0f, 4.0f, 7.0f, 3.0f, 3.0f,
                                     12.0f, 8.0f, 13.0f, 6.0f, 3.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_1channel_dilation)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{2, 2};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 1, 2, 2};
    const std::vector<float> inputs{2.0f, 3.0f,
                                    4.0f, 3.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     1.0f, 1.0f, 1.0f,
                                     3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 6, 6};
    const std::vector<float> outputs{2.f, 3.f, 4.f, 6.f, 6.f, 9.f,
                                     4.f, 3.f, 8.f, 6.f, 12.f, 9.f,
                                     2.f, 3.f, 2.f, 3.f, 2.f, 3.f,
                                     4.f, 3.f, 4.f, 3.f, 4.f, 3.f,
                                     6.f, 9.f, 4.f, 6.f, 2.f, 3.f,
                                     12.f, 9.f, 8.f, 6.f, 4.f, 3.f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_1channel_padding_strides_dilation)
{
    const Strides strides{2, 2};
    const CoordinateDiff padding{2, 2};
    const Strides dilations{2, 2};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 1, 3, 3};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f,
                                    7.0f, 5.0f, 3.0f,
                                    2.0f, 4.0f, 6.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                     1.0f, 1.0f, 1.0f,
                                     3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{1, 1, 5, 5};
    const std::vector<float> outputs{23.0f, 0.0f, 43.0f, 0.0f, 29.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                     31.0f, 0.0f, 57.0f, 0.0f, 45.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                     35.0f, 0.0f, 38.0f, 0.0f, 21.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_2channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 1, 2, 2};
    const std::vector<float> inputs{1.0f, 3.0f,
                                    7.0f, 5.0f};

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

    const Shape outputs_shape{1, 2, 4, 4};
    const std::vector<float> outputs{
                                    // channel 1
                                    5.0f, 18.0f, 14.0f, 15.0f,
                                    36.0f, 52.0f, 60.0f, 28.0f,
                                    11.0f, 40.0f, 32.0f, 17.0f,
                                    28.0f, 34.0f, 38.0f, 20.0f,
                                    // channel 2
                                    -5.0f, -12.0f, 14.0f, 15.0f,
                                    -34.0f, -4.0f, 42.0f, 28.0f,
                                    11.0f, -2.0f, -6.0f, -7.0f,
                                    28.0f, 34.0f, -18.0f, -20.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_1batch_2filter)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 2, 2, 2};
    const std::vector<float> inputs{
                                    // channel 1
                                    1.0f, 3.0f,
                                    7.0f, 5.0f,
                                    // channel 2
                                    2.0f, 4.0f,
                                    8.0f, 6.0f};

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

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{
                                     -5.0f, 4.0f, 36.0f, 35.0f,
                                     -2.0f, 44.0f, 108.0f, 62.0f,
                                     27.0f, 42.0f, 22.0f, 7.0f,
                                     60.0f, 74.0f, 18.0f, -4.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_2batch_2filter)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{1, 2, 1, 1};
    const std::vector<float> inputs{
                                    // channel 1
                                    2.0f,
                                    // channel 2
                                    3.0f};

    const Shape filter_shape{2, 2, 2, 2};
    const std::vector<float> filters{
                                    // batch 0
                                    // channel 1
                                    5.0f, 3.0f,
                                    1.0f, 3.0f,
                                    // channel 2
                                   -5.0f, 3.0f,
                                    1.0f, -3.0f,
                                    // batch 1
                                    // channel 1
                                    5.0f, 3.0f,
                                    1.0f, 3.0f,
                                    // channel 2
                                   -5.0f, 3.0f,
                                    1.0f, -3.0f};

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{
                                     25.0f, 15.0f, 5.0f, 15.0f, -25.0f, 15.0f, 5.0f, -15.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_2D_2batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};
    const CoordinateDiff output_padding{0, 0};

    const Shape inputs_shape{2, 1, 2, 2};
    const std::vector<float> inputs{
                                    // batch 1
                                    1.0f, 3.0f, 
                                    1.0f, 3.0f,
                                    // batch 2
                                    -1.0f, 3.0f,
                                    1.0f, 3.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{-5.0f, 3.0f, 5.0f,
                                    1.0f, -3.0f, 1.0f,
                                    4.0f, 2.0f, -4.0f};

    const Shape outputs_shape{2, 1, 4, 4};
    const std::vector<float> outputs{
                                    // batch 1
                                    -5.0f, -12.0f, 14.0f, 15.0f,
                                    -4.0f, -12.0f, 6.0f, 18.0f,
                                    5.0f, 14.0f, -6.0f, -9.0f,
                                    4.0f, 14.0f, 2.0f, -12.0f,
                                    // batch 2
                                    5.0f, -18.0f, 4.0f, 15.0f,
                                    -6.0f, -6.0f, 4.0f, 18.0f,
                                    -3.0f, 10.0f, 2.0f, -9.0f,
                                    4.0f, 14.0f, 2.0f, -12.0f};


    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

// --------------------- 3D convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_1channel)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{0, 0, 0};

    const Shape inputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> inputs{
                                    // depth: 1
                                    15.0f, 3.0f,
                                    21.0f, 10.0f,
                                    // depth: 2
                                    10.0f, 13.0f,
                                    11.0f, 17.0f};

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
                                    15.0f, 33.0f, 51.0f, 9.0f,
                                    21.0f, 67.0f, 86.0f, 30.0f,
                                    30.0f, 42.0f, 43.0f, 6.0f,
                                    42.0f, 41.0f, 52.0f, 20.0f,
                                    // depth: 2
                                    25.0f, 66.0f, 107.0f, 48.0f,
                                    32.0f, 116.0f, 166.0f, 81.0f,
                                    50.0f, 89.0f, 93.0f, 32.0f,
                                    64.0f, 86.0f, 91.0f, 54.0f,
                                    // depth: 3
                                    25.0f, 66.0f, 107.0f, 48.0f,
                                    32.0f, 116.0f, 166.0f, 81.0f,
                                    50.0f, 89.0f, 93.0f, 32.0f,
                                    64.0f, 86.0f, 91.0f, 54.0f,
                                    // depth: 4
                                    10.0f, 33.0f, 56.0f, 39.0f,
                                    11.0f, 49.0f, 80.0f, 51.0f,
                                    20.0f, 47.0f, 50.0f, 26.0f,
                                    22.0f, 45.0f, 39.0f, 34.0f
                                    };

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_1channel_output_padding)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{1, 1, 1};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{1, 1, 1};

    const Shape inputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> inputs{
                                    // depth: 1
                                    15.0f, 3.0f,
                                    21.0f, 10.0f,
                                    // depth: 2
                                    10.0f, 13.0f,
                                    11.0f, 17.0f};

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

    const Shape outputs_shape{1, 1, 3, 3, 3};
    const std::vector<float> outputs{
                                    // depth: 1
                                    116.0f, 166.0f, 81.0f,
                                    89.0f, 93.0f, 32.0f,
                                    86.0f, 91.0f, 54.0f,
                                    // depth: 2
                                    116.0f, 166.0f, 81.0f,
                                    89.0f, 93.0f, 32.0f,
                                    86.0f, 91.0f, 54.0f,
                                    // depth: 3
                                    49.0f, 80.0f, 51.0f,
                                    47.0f, 50.0f, 26.0f,
                                    45.0f, 39.0f, 34.0f
                                    };

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_1channel_padding)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{1, 1, 1};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{0, 0, 0};

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
                                     12.0f, 30.0f, 36.0f, 24.0f,
                                     26.0f, 42.0f, 42.0f, 30.0f,
                                     34.0f, 56.0f, 54.0f, 50.0f,
                                     14.0f, 18.0f, 24.0f, 16.0f,
                                     // depth: 2
                                     18.0f, 45.0f, 54.0f, 36.0f,
                                     39.0f, 63.0f, 63.0f, 45.0f,
                                     51.0f, 84.0f, 81.0f, 75.0f,
                                     21.0f, 27.0f, 36.0f, 24.0f,
                                     // depth: 3
                                     18.0f, 45.0f, 54.0f, 36.0f,
                                     39.0f, 63.0f, 63.0f, 45.0f,
                                     51.0f, 84.0f, 81.0f, 75.0f,
                                     21.0f, 27.0f, 36.0f, 24.0f,
                                     // depth: 4
                                     12.0f, 30.0f, 36.0f, 24.0f,
                                     26.0f, 42.0f, 42.0f, 30.0f,
                                     34.0f, 56.0f, 54.0f, 50.0f,
                                     14.0f, 18.0f, 24.0f, 16.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_1channel_stride)
{
    const Strides strides{2, 2, 2};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{0, 0, 0};

    const Shape inputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> inputs{
                                    // depth: 1
                                    15.0f, 3.0f,
                                    21.0f, 10.0f,
                                    // depth: 2
                                    10.0f, 13.0f,
                                    11.0f, 17.0f};

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

    const Shape outputs_shape{1, 1, 5, 5, 5};
    const std::vector<float> outputs{
                                    // depth: 1
                                    15.0f, 30.0f, 48.0f, 6.0f, 9.0f,
                                    0.0f, 15.0f, 0.0f, 3.0f, 0.0f,
                                    51.0f, 57.0f, 109.0f, 23.0f, 36.0f,
                                    0.0f, 21.0f, 0.0f, 10.0f, 0.0f,
                                    42.0f, 21.0f, 62.0f, 10.0f, 20.0f,
                                    // depth: 2
                                    15.0f, 30.0f, 48.0f, 6.0f, 9.0f,
                                    0.0f, 15.0f, 0.0f, 3.0f, 0.0f,
                                    51.0f, 57.0f, 109.0f, 23.0f, 36.0f,
                                    0.0f, 21.0f, 0.0f, 10.0f, 0.0f,
                                    42.0f, 21.0f, 62.0f, 10.0f, 20.0f,
                                    // depth: 3
                                    25.0f, 50.0f, 91.0f, 32.0f, 48.0f,
                                    0.0f, 25.0f, 0.0f, 16.0f, 0.0f,
                                    82.0f, 89.0f, 205.0f, 70.0f, 113.0f,
                                    0.0f, 32.0f, 0.0f, 27.0f, 0.0f,
                                    64.0f, 32.0f, 118.0f, 27.0f, 54.0f,
                                    // depth: 4
                                    10.0f, 20.0f, 43.0f, 26.0f, 39.0f,
                                    0.0f, 10.0f, 0.0f, 13.0f, 0.0f,
                                    31.0f, 32.0f, 96.0f, 47.0f, 77.0f,
                                    0.0f, 11.0f, 0.0f, 17.0f, 0.0f,
                                    22.0f, 11.0f, 56.0f, 17.0f, 34.0f,
                                    // depth: 5
                                    10.0f, 20.0f, 43.0f, 26.0f, 39.0f,
                                    0.0f, 10.0f, 0.0f, 13.0f, 0.0f,
                                    31.0f, 32.0f, 96.0f, 47.0f, 77.0f,
                                    0.0f, 11.0f, 0.0f, 17.0f, 0.0f,
                                    22.0f, 11.0f, 56.0f, 17.0f, 34.0f
                                    };

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_1channel_padding_strides_dilation)
{
    const Strides strides{2, 2, 2};
    const CoordinateDiff padding{2, 2, 2};
    const Strides dilations{2, 2, 2};
    const CoordinateDiff output_padding{0, 0, 0};

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

    const Shape outputs_shape{1, 1, 7, 7, 7};
    const std::vector<float> outputs{
                                    // depth: 1
                                    12.0f, 0.0f, 30.0f, 0.0f, 36.0f, 0.0f, 24.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    26.0f, 0.0f, 42.0f, 0.0f, 42.0f, 0.0f, 30.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    34.0f, 0.0f, 56.0f, 0.0f, 54.0f, 0.0f, 50.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    14.0f, 0.0f, 18.0f, 0.0f, 24.0f, 0.0f, 16.0f,
                                    // depth: 2
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    // depth: 3
                                    18.0f, 0.0f, 45.0f, 0.0f, 54.0f, 0.0f, 36.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    39.0f, 0.0f, 63.0f, 0.0f, 63.0f, 0.0f, 45.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    51.0f, 0.0f, 84.0f, 0.0f, 81.0f, 0.0f, 75.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    21.0f, 0.0f, 27.0f, 0.0f, 36.0f, 0.0f, 24.0f,
                                    // depth: 4
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    // depth: 5
                                    18.0f, 0.0f, 45.0f, 0.0f, 54.0f, 0.0f, 36.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    39.0f, 0.0f, 63.0f, 0.0f, 63.0f, 0.0f, 45.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    51.0f, 0.0f, 84.0f, 0.0f, 81.0f, 0.0f, 75.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    21.0f, 0.0f, 27.0f, 0.0f, 36.0f, 0.0f, 24.0f,
                                    // depth: 6
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    // depth: 7
                                    12.0f, 0.0f, 30.0f, 0.0f, 36.0f, 0.0f, 24.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    26.0f, 0.0f, 42.0f, 0.0f, 42.0f, 0.0f, 30.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    34.0f, 0.0f, 56.0f, 0.0f, 54.0f, 0.0f, 50.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    14.0f, 0.0f, 18.0f, 0.0f, 24.0f, 0.0f, 16.0f
                                    };

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_2channel)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{0, 0, 0};

    const Shape inputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> inputs{
                                    // depth: 1
                                    1.0f, 8.0f,
                                    1.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 7.0f,
                                    3.0f, 8.0f};

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

    const Shape outputs_shape{1, 2, 4, 4, 4};
    const std::vector<float> outputs{
                                    // -- channel 1 --
                                    // depth: 1
                                    1.0f, 10.0f, 19.0f, 24.0f,
                                    1.0f, 6.0f, 17.0f, 9.0f,
                                    2.0f, 18.0f, 13.0f, 16.0f,
                                    2.0f, 7.0f, 5.0f, 6.0f,
                                    // depth: 2
                                    2.0f, 19.0f, 36.0f, 45.0f,
                                    4.0f, 21.0f, 49.0f, 33.0f,
                                    4.0f, 36.0f, 30.0f, 30.0f,
                                    8.0f, 26.0f, 19.0f, 22.0f,
                                    // depth: 3
                                    2.0f, 19.0f, 36.0f, 45.0f,
                                    4.0f, 21.0f, 49.0f, 33.0f,
                                    4.0f, 36.0f, 30.0f, 30.0f,
                                    8.0f, 26.0f, 19.0f, 22.0f,
                                    // depth: 4
                                    1.0f, 9.0f, 17.0f, 21.0f,
                                    3.0f, 15.0f, 32.0f, 24.0f,
                                    2.0f, 18.0f, 17.0f, 14.0f,
                                    6.0f, 19.0f, 14.0f, 16.0f,
                                    // -- channel 2 --
                                    // depth: 1
                                    1.0f, 10.0f, 19.0f, 24.0f,
                                    1.0f, 6.0f, 17.0f, 9.0f,
                                    2.0f, 18.0f, 13.0f, 16.0f,
                                    2.0f, 7.0f, 5.0f, 6.0f,
                                    // depth: 2
                                    2.0f, 19.0f, 36.0f, 45.0f,
                                    4.0f, 21.0f, 49.0f, 33.0f,
                                    4.0f, 36.0f, 30.0f, 30.0f,
                                    8.0f, 26.0f, 19.0f, 22.0f,
                                    // depth: 3
                                    2.0f, 19.0f, 36.0f, 45.0f,
                                    4.0f, 21.0f, 49.0f, 33.0f,
                                    4.0f, 36.0f, 30.0f, 30.0f,
                                    8.0f, 26.0f, 19.0f, 22.0f,
                                    // depth: 4
                                    1.0f, 9.0f, 17.0f, 21.0f,
                                    3.0f, 15.0f, 32.0f, 24.0f,
                                    2.0f, 18.0f, 17.0f, 14.0f,
                                    6.0f, 19.0f, 14.0f, 16.0f
                                    };

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_1batch_2filter)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{0, 0, 0};

    const Shape inputs_shape{1, 2, 2, 2, 2};
    const std::vector<float> inputs{
                                    // -- in 1 --
                                    // depth: 1
                                    1.0f, 3.0f,
                                    2.0f, 5.0f,
                                    // depth: 2
                                    1.0f, 0.0f,
                                    3.0f, 6.0f,
                                    // -- in 2 --
                                    // depth: 1
                                    1.0f, 3.0f,
                                    2.0f, 5.0f,
                                    // depth: 2
                                    3.0f, 0.0f,
                                    1.0f, 8.0f};

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

    const Shape outputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> outputs{
                                     // depth: 1
                                     2.0f, 10.0f, 18.0f, 18.0f,
                                     4.0f, 20.0f, 38.0f, 30.0f,
                                     4.0f, 18.0f, 20.0f, 12.0f,
                                     8.0f, 24.0f, 18.0f, 20.0f,
                                     // depth: 2
                                     6.0f, 18.0f, 30.0f, 18.0f,
                                     8.0f, 46.0f, 78.0f, 72.0f,
                                     12.0f, 26.0f, 42.0f, 12.0f,
                                     16.0f, 56.0f, 40.0f, 48.0f,
                                     // depth: 3
                                     6.0f, 18.0f, 30.0f, 18.0f,
                                     8.0f, 46.0f, 78.0f, 72.0f,
                                     12.0f, 26.0f, 42.0f, 12.0f,
                                     16.0f, 56.0f, 40.0f, 48.0f,
                                     // depth: 4
                                     4.0f, 8.0f, 12.0f, 0.0f,
                                     4.0f, 26.0f, 40.0f, 42.0f,
                                     8.0f, 8.0f, 22.0f, 0.0f,
                                     8.0f, 32.0f, 22.0f, 28.0f
                                     };

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_backprop_3D_2batch_1channel)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};
    const CoordinateDiff output_padding{0, 0, 0};

    const Shape inputs_shape{2, 1, 2, 2, 2};
    const std::vector<float> inputs{
                                    // -- batch 1 --
                                    // depth: 1
                                    1.0f, 3.0f,
                                    2.0f, 5.0f,
                                    // depth: 2
                                    1.0f, 0.0f,
                                    6.0f, 4.0f,
                                    // -- batch 2 --
                                    // depth: 1
                                    1.0f, 5.0f,
                                    2.0f, 8.0f,
                                    // depth: 2
                                    2.0f, 1.0f,
                                    0.0f, 5.0f};
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

    const Shape outputs_shape{2, 1, 4, 4, 4};
    const std::vector<float> outputs{
                                     // -- batch 1 --
                                     // depth: 1
                                     1.0f, 5.0f, 9.0f, 9.0f,
                                     2.0f, 10.0f, 19.0f, 15.0f,
                                     2.0f, 9.0f, 10.0f, 6.0f,
                                     4.0f, 12.0f, 9.0f, 10.0f,
                                     // depth: 2
                                     2.0f, 7.0f, 12.0f, 9.0f,
                                     8.0f, 27.0f, 45.0f, 27.0f,
                                     4.0f, 16.0f, 16.0f, 6.0f,
                                     16.0f, 26.0f, 25.0f, 18.0f,
                                     // depth: 3
                                     2.0f, 7.0f, 12.0f, 9.0f,
                                     8.0f, 27.0f, 45.0f, 27.0f,
                                     4.0f, 16.0f, 16.0f, 6.0f,
                                     16.0f, 26.0f, 25.0f, 18.0f,
                                     // depth: 4
                                     1.0f, 2.0f, 3.0f, 0.0f,
                                     6.0f, 17.0f, 26.0f, 12.0f,
                                     2.0f, 7.0f, 6.0f, 0.0f,
                                     12.0f, 14.0f, 16.0f, 8.0f,
                                     // -- batch 2 --
                                     // depth: 1
                                     1.0f, 7.0f, 13.0f, 15.0f,
                                     2.0f, 13.0f, 27.0f, 24.0f,
                                     2.0f, 13.0f, 15.0f, 10.0f,
                                     4.0f, 18.0f, 12.0f, 16.0f,
                                     // depth: 2
                                     3.0f, 12.0f, 21.0f, 18.0f,
                                     2.0f, 20.0f, 38.0f, 39.0f,
                                     6.0f, 17.0f, 25.0f, 12.0f,
                                     4.0f, 28.0f, 17.0f, 26.0f,
                                     // depth: 3
                                     3.0f, 12.0f, 21.0f, 18.0f,
                                     2.0f, 20.0f, 38.0f, 39.0f,
                                     6.0f, 17.0f, 25.0f, 12.0f,
                                     4.0f, 28.0f, 17.0f, 26.0f,
                                     // depth: 4
                                     2.0f, 5.0f, 8.0f, 3.0f,
                                     0.0f, 7.0f, 11.0f, 15.0f,
                                     4.0f, 4.0f, 10.0f, 2.0f,
                                     0.0f, 10.0f, 5.0f, 10.0f};

    ConvolutionBackpropTest(inputs, inputs_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations, output_padding);
}
