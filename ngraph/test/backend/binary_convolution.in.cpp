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
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename T_IN, typename T_KERN>
static void BinaryConvolutionTest(const std::vector<T_IN>& inputs,
                                  const Shape inputs_shape,
                                  const std::vector<T_KERN>& filters,
                                  const Shape filter_shape,
                                  const std::vector<T_IN>& outputs,
                                  const Shape outputs_shape,
                                  const Strides& strides,
                                  const CoordinateDiff& padding,
                                  const Strides& dilations)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};
    float pad_value = 0;

    auto inputs_param = make_shared<op::Parameter>(element::from<T_IN>(), inputs_shape);
    auto filters_const = make_shared<op::Constant>(element::u1, filter_shape, &filters[0]);
    auto bin_conv = make_shared<op::v1::BinaryConvolution>(
        inputs_param,
        filters_const,
        strides,
        pads_begin,
        pads_end,
        dilations,
        op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
        pad_value,
        auto_pad);
    auto f = make_shared<Function>(bin_conv, ParameterVector{inputs_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<T_IN>(inputs);
    test_case.add_expected_output<T_IN>(outputs_shape, outputs);
    test_case.run();
}

// --------------------- 1D convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, bin_convolution_1D_1batch_1channel_no_padding)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 5};
    const std::vector<float> inputs{1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    const Shape filter_shape{1, 1, 3};
    const std::vector<uint8_t> filters{160}; // filters 1D: {1.0f, 0.0f, 1.0f}

    const Shape outputs_shape{1, 1, 3};
    const std::vector<float> outputs{1.0f, 1.0f, -3.0f};

    BinaryConvolutionTest(inputs,
                          inputs_shape,
                          filters,
                          filter_shape,
                          outputs,
                          outputs_shape,
                          strides,
                          padding,
                          dilations);
}

// --------------------- 2D convolution ------------------------------------------
// clang-format off
NGRAPH_TEST(${BACKEND_NAME}, bin_convolution_2D_1batch_1channel_no_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 0.0f, 0.0f, 1.0f,
                                    1.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 1.0f,
                                    1.0f, 0.0f, 1.0f, 1.0f};

    const Shape filter_shape{1, 1, 3, 3};
    //filters 2D: {1.0f, 0.0f, 1.0f,
    //             0.0f, 1.0f, 0.0f,
    //             1.0f, 0.0f, 1.0f};
    const std::vector<uint8_t> filters{170, 128};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{1.0f, 1.0f,
                                     3.0f, -1.0f};

    BinaryConvolutionTest(inputs,
                          inputs_shape,
                          filters,
                          filter_shape,
                          outputs,
                          outputs_shape,
                          strides,
                          padding,
                          dilations);
}
// clang-format on
