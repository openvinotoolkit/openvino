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


// --------------------- 2D convolution ------------------------------------------
// clang-format off
NGRAPH_TEST(${BACKEND_NAME}, binary_convolution_2D_1batch_1channel)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<bool> inputs{1, 0, 0, 1,
                                  0, 1, 1, 0,
                                  0, 0, 0, 0,
                                  1, 1, 1, 1};

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<bool> filters{1, 0, 1,
                                   0, 1, 0,
                                   0, 1, 1};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{0, 0,
                                   0, 0};

    
    const op::PadType auto_pad{op::PadType::VALID};
    float pad_value = 0;

    auto inputs_param = make_shared<op::Parameter>(element::boolean, inputs_shape);
    auto filters_param = make_shared<op::Parameter>(element::boolean, filter_shape);
    auto conv = make_shared<op::v1::BinaryConvolution>(
        inputs_param,
        filters_param,
        strides,
        pads_begin,
        pads_end,
        dilations,
        op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
        pad_value,
        auto_pad);
    auto f = make_shared<Function>(conv, ParameterVector{inputs_param, filters_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input(inputs);
    test_case.add_input(filters);
    test_case.add_expected_output(outputs_shape, outputs);
    test_case.run();
}
// clang-format on
