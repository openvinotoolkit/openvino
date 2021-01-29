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
/*
NGRAPH_TEST(${BACKEND_NAME}, deformable_convolution_1D_1batch_1channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 6};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f};

    const Shape deformable_values_shape{1, 1, 6};
    const std::vector<float> deformable_values{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4};
    const std::vector<float> outputs{5.0f, 6.0f, 7.0f, 2.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}
*/

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
    
    const Shape deformable_values_shape{1, 1, 3, 3};
    const std::vector<float> deformable_values{0.1f, 0.3f, 0.2f,
                                               0.6f, 0.5f, 0.1f,
                                               0.3f, 0.1f, 0.6f};

    const Shape outputs_shape{1, 1, 4, 4};
    const std::vector<float> outputs{18.0f, 28.0f, 20.0f, 14.0f,
                                     28.0f, 40.0f, 64.0f, 35.0f,
                                     51.0f, 53.0f, 35.0f, 23.0f,
                                     24.0f, 32.0f, 44.0f, 24.0f};

    DeformableConvolutionTest(inputs, inputs_shape, deformable_values, deformable_values_shape, filters, filter_shape, outputs, outputs_shape,
                    strides, padding, dilations);
}