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

static void DeformableConvolutionTest(const std::vector<float>& inputs,
                            const Shape inputs_shape,
                            const std::vector<float>& deformable_values,
                            const Shape deformable_values_shape,
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
    auto deformable_values_param = make_shared<op::Parameter>(element::i32, deformable_values_shape);
    auto filters_param = make_shared<op::Parameter>(element::f32, filter_shape);
    auto conv = make_shared<op::v1::DeformableConvolution>(
        inputs_param, deformable_values_param, filters_param, strides, pads_begin, pads_end, dilations, auto_pad);
    auto f = make_shared<Function>(conv, ParameterVector{inputs_param, deformable_values_param, filters_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs);
    test_case.add_input<float>(deformable_values);
    test_case.add_input<float>(filters);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_1batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 7.0f, 7.0f,
                                    7.0f, 6.0f, 3.0f, 1.0f,
                                    4.0f, 4.0f, 2.0f, 8.0f,
                                    1.0f, 1.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape deformable_values_shape{1, 18, 2, 2};
    const std::vector<float> deformable_values(1 * 18 * 2 * 2, 0);

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{56.0f, 65.0f,
                                     38.0f, 24.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_deformable_convolution_2D_1batch_1channel_1padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{1, 1};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 3, 3};
    const std::vector<float> inputs{1.0f, 3.0f, 5.0f,
                                    7.0f, 5.0f, 3.0f,
                                    1.0f, 3.0f, 5.0f};

    const Shape filter_shape{1, 1, 2, 2};
    const std::vector<float> filters{1.0f, 2.0f,
                                     0.0f, 1.0f};
    
    const Shape deformable_values_shape{1, 8, 5, 5};
    const std::vector<float> deformable_values(1 * 8 * 5 * 5, 0);

                                               

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{1.0f, 3.0f, 5.0f, 0.0f,
                                     9.0f, 12.0f, 16.0f, 5.0f,
                                     15.0f, 20.0f, 16.0f, 3.0f,
                                     2.0f, 7.0f, 13.0f, 5.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_1batch_1channel_stride)
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
    
    const Shape deformable_values_shape{1, 18, 2, 2};
    const std::vector<float> deformable_values(1 * 18 * 2 * 2, 0);

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{57.0f, 94.0f,
                                     66.0f, 102.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_1batch_1channel_dilation)
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

    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values(1 * 18 * 3 * 3, 0);

    const Shape outputs_shape{1, 1, 3, 3};
    const std::vector<float> outputs{78.0f, 106.0f, 134.0f,
                                     44.0f, 16.0f, -12.0f,
                                     80.0f, 84.0f, 88.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_1batch_1channel_padding_strides_dilation)
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

    const Shape deformable_values_shape{1, 18, 4, 4};
    const std::vector<float> deformable_values(1 * 18 * 4 * 4, 0);                                

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{15.0f, 38.0f, 70.0f, 66.0f,
                                    33.0f, 78.0f, 134.0f, 103.0f,
                                    40.0f, 80.0f, 88.0f, 58.0f,
                                    30.0f, 56.0f, 72.0f, 34.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_1batch_2channel)
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

    const Shape deformable_values_shape{1, 18, 2, 2};
    const std::vector<float> deformable_values(1 * 18 * 2 * 2, 0);
    
    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{142.0f, 102.0f,
                                     94.0f, 160.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_1batch_2filter)
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

    const Shape deformable_values_shape{1, 18, 2, 2};
    const std::vector<float> deformable_values(1 * 18 * 2 * 2);

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // channel 1
                                     104.0f, 140.0f,
                                     145.0f, 109.0f,
                                     // channel 2
                                     16.0f, 28.0f,
                                     19.0f, 7.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_deformable_values_zero_2D_2batch_1channel)
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
    
    const Shape deformable_values_shape{2, 18, 2, 2};
    const std::vector<float> deformable_values(2 * 18 * 2 * 2, 0);

    const Shape outputs_shape{2, 1, 2, 2};
    const std::vector<float> outputs{
                                    // batch 1
                                    15.0f, -15.0f,
                                    23.0f, 2.0f,
                                    // batch 2
                                    -1.0f, -15.0f,
                                    -5.0f, 6.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1group_1batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 6, 6};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f,
                                     2.0f, 1.0f, 1.0f,
                                     2.0f, 3.0f, 1.0f};

    const Shape deformable_values_shape{1, 18, 4, 4};
    const std::vector<float> deformable_values(1 * 18 * 4 * 4, 0);

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{27.0f, 30.0f, 21.0f, 10.0f,
                                     27.0f, 30.0f, 21.0f, 10.0f,
                                     27.0f, 30.0f, 21.0f, 10.0f,
                                     27.0f, 30.0f, 21.0f, 10.0f,};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_2group_1batch_2channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 2, 5, 5};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    };

    const Shape filter_shape{2, 2, 3, 3};
    const std::vector<float> filters{1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     3.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     3.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     3.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     3.0f, 0.0f, 1.0f};
    
    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values(1 * 18 * 3 * 3, 0);

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{44.0f, 62.0f, 72.0f, 
                                     44.0f, 62.0f, 72.0f, 
                                     44.0f, 62.0f, 72.0f, 
                                     44.0f, 62.0f, 72.0f,  
                                     44.0f, 62.0f, 72.0f,  
                                     44.0f, 62.0f, 72.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_2group_1batch_2_filters_2channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 2, 5, 5};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 2.0f, 1.0f,
                                    0.0f, 1.0f, 2.0f, 2.0f, 3.0f,
                                    -1.0f, -3.0f, -3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 2.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 2.0f, 2.0f, 3.0f,
                                    -1.0f, -3.0f, -3.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 2.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 2.0f, 2.0f, 3.0f};

    const Shape filter_shape{2, 2, 3, 3};
    const std::vector<float> filters{1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     -3.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     -3.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     -3.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 3.0f,
                                     3.0f, 0.0f, 1.0f,
                                     -3.0f, 0.0f, 1.0f};
    
    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values(1 * 18 * 3 * 3, 0);

    const Shape outputs_shape{1, 2, 3, 3};
    const std::vector<float> outputs{12.0f, 18.0f, 26.0f, 
                                     -2.0f, 6.0f, 14.0f,
                                     12.0f, 26.0f, 33.0f,
                                     12.0f, 18.0f, 26.0f,
                                     -2.0f, 6.0f, 14.0f,
                                     12.0f, 26.0f, 33.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

// --------------------- 2D deformable convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 7.0f, 7.0f,
                                    7.0f, 6.0f, 3.0f, 1.0f,
                                    4.0f, 4.0f, 2.0f, 8.0f,
                                    1.0f, 1.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values{1.0f, 1.0f, 0.0f,
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

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{33.0f, 4.0f,
                                     7.0f, 9.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_1channel_padding)
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
    
    const Shape deformable_values_shape{1, 18, 4, 4};
    const std::vector<float> deformable_values( 1 * 18 * 4 * 4, 0);

                                               

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{18.0f, 28.0f, 20.0f, 14.0f,
                                     28.0f, 47.0f, 67.0f, 40.0f,
                                     51.0f, 60.0f, 40.0f, 23.0f,
                                     24.0f, 34.0f, 44.0f, 24.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_1channel_stride)
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
    
    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values{1.0f, 1.0f, 0.0f,
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

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{21.0f, 19.0f,
                                     6.0f, 34.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_1channel_dilation)
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

    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values{1.0f, 1.0f, 0.0f,
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
    const std::vector<float> outputs{0.0f, 0.0f, 134.0f,
                                     44.0f, 0.0f, -12.0f,
                                     0.0f, 84.0f, 0.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_1channel_padding_strides_dilation)
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

    const Shape deformable_values_shape{1, 18, 4, 4};
    const std::vector<float> deformable_values(1 * 18 * 4 * 4, 0);                                

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{15.0f, 38.0f, 70.0f, 66.0f,
                                    33.0f, 78.0f, 134.0f, 103.0f,
                                    40.0f, 80.0f, 88.0f, 58.0f,
                                    30.0f, 56.0f, 72.0f, 34.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_2channel)
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

    const Shape deformable_values_shape{1, 18, 2, 2};
    const std::vector<float> deformable_values(1 * 18 * 2 * 2, 0);
    
    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{142.0f, 102.0f,
                                     94.0f, 160.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_1batch_2filter)
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

    const Shape deformable_values_shape{1, 18, 3, 3};
    const std::vector<float> deformable_values{1.0f, 1.0f, 0.0f,
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

    const Shape outputs_shape{1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // channel 1
                                     33.0f, 17.0f,
                                     41.0f, 33.0f,
                                     // channel 2
                                     33.0f, 17.0f,
                                     -29.0f, 33.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_2D_2batch_1channel)
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
    
    const Shape deformable_values_shape{2, 18, 3, 3};
    const std::vector<float> deformable_values{
                                               // batch 1
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
                                               1.0f, 1.0f, 0.0f,
                                               0.0f, 2.0f, 0.0f,
                                               1.0f, 0.0f, 1.0f,
                                               1.0f, 1.0f, 0.0f,
                                               0.0f, 2.0f, 0.0f,
                                               1.0f, 0.0f, 1.0f,
                                               // batch 2
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
                                               1.0f, 1.0f, 0.0f,
                                               0.0f, 2.0f, 0.0f,
                                               1.0f, 0.0f, 1.0f,
                                               1.0f, 2.0f, 0.0f,
                                               0.0f, 2.0f, 0.0f,
                                               1.0f, 0.0f, 1.0f};

    const Shape outputs_shape{2, 1, 2, 2};
    const std::vector<float> outputs{
                                    // batch 1
                                    18.0f, 5.0f,
                                    -4.0f, 17.0f,
                                    // batch 2
                                    2.0f, 5.0f,
                                    -4.0f, -1.0f};


    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters,
                              filter_shape, outputs, outputs_shape,strides, padding, dilations);
}
