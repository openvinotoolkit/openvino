// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/execute_tools.hpp"
#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename T_IN, typename T_KERN>
static void BinaryConvolutionTest(const std::vector<T_IN>& inputs,
                                  const Shape inputs_shape,
                                  const std::vector<T_KERN>& filters,
                                  const Shape filters_shape,
                                  const std::vector<T_IN>& outputs,
                                  const Shape outputs_shape,
                                  const Strides& strides,
                                  const CoordinateDiff& padding,
                                  const Strides& dilations) {
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};
    float pad_value = 0;

    auto inputs_param = make_shared<op::Parameter>(element::from<T_IN>(), inputs_shape);
    auto filters_const = make_shared<op::Constant>(element::u1, filters_shape, &filters[0]);
    auto bin_conv =
        make_shared<op::v1::BinaryConvolution>(inputs_param,
                                               filters_const,
                                               strides,
                                               pads_begin,
                                               pads_end,
                                               dilations,
                                               op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                               pad_value,
                                               auto_pad);
    auto f = make_shared<Function>(bin_conv, ParameterVector{inputs_param});
    auto test_case = test::TestCase(f);
    test_case.add_input(inputs_shape, inputs);
    test_case.add_expected_output(outputs_shape, outputs);
    test_case.run();
}

template <typename T_IN>
static void ConvolutionTest(const std::vector<T_IN>& inputs,
                            const Shape inputs_shape,
                            const std::vector<T_IN>& filters,
                            const Shape filters_shape,
                            const std::vector<T_IN>& outputs,
                            const Shape outputs_shape,
                            const Strides& strides,
                            const CoordinateDiff& padding,
                            const Strides& dilations) {
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};

    auto inputs_param = make_shared<op::Parameter>(element::from<T_IN>(), inputs_shape);
    auto filters_param = make_shared<op::Parameter>(element::from<T_IN>(), filters_shape);
    auto conv = make_shared<op::v1::Convolution>(inputs_param,
                                                 filters_param,
                                                 strides,
                                                 pads_begin,
                                                 pads_end,
                                                 dilations,
                                                 auto_pad);
    auto f = make_shared<Function>(conv, ParameterVector{inputs_param, filters_param});

    auto test_case = test::TestCase(f);
    test_case.add_input(inputs_shape, inputs);
    test_case.add_input(filters_shape, filters);
    test_case.add_expected_output(outputs_shape, outputs);
    test_case.run();
}

// --------------------- 1D convolution ------------------------------------------
TEST(op_eval, bin_convolution_1D_1batch_1channel_no_padding) {
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 5};
    const std::vector<float> inputs_conv{1.0f, -1.0f, -1.0f, 1.0f, -1.0f};
    const std::vector<float> inputs_bin_conv{1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    const Shape filters_shape{1, 1, 3};
    const std::vector<float> filters_conv{1.0f, -1.0f, 1.0f};
    const std::vector<uint8_t> filters_bin_conv{0xA0};  // 1010 0000

    const Shape outputs_shape{1, 1, 3};
    const std::vector<float> outputs{1.0f, 1.0f, -3.0f};

    BinaryConvolutionTest(inputs_bin_conv,
                          inputs_shape,
                          filters_bin_conv,
                          filters_shape,
                          outputs,
                          outputs_shape,
                          strides,
                          padding,
                          dilations);

    ConvolutionTest(inputs_conv,
                    inputs_shape,
                    filters_conv,
                    filters_shape,
                    outputs,
                    outputs_shape,
                    strides,
                    padding,
                    dilations);
}

// --------------------- 3D convolution ------------------------------------------
// clang-format off
NGRAPH_TEST(op_eval, bin_convolution_3D_1batch_1channel_no_padding)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 1, 4, 4, 4};
    const std::vector<float> inputs_conv{
                                        // depth: 1
                                        1.0f, -1.0f, -1.0f, 1.0f,
                                        -1.0f, 1.0f, -1.0f, 1.0f,
                                        1.0f, 1.0f, -1.0f, 1.0f,
                                        1.0f, -1.0f, 1.0f, -1.0f,
                                        // depth: 2
                                        -1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, -1.0f, -1.0f, 1.0f,
                                        -1.0f, 1.0f, -1.0f, 1.0f,
                                        1.0f, -1.0f, 1.0f, -1.0f,
                                        // depth: 3
                                        1.0f, 1.0f, 1.0f, -1.0f,
                                        -1.0f, 1.0f, -1.0f, 1.0f,
                                        1.0f, 1.0f, -1.0f, 1.0f,
                                        1.0f, -1.0f, 1.0f, -1.0f,
                                        // depth: 4
                                        1.0f, -1.0f, 1.0f, -1.0f,
                                        1.0f, 1.0f, -1.0f, 1.0f,
                                        -1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, -1.0f, 1.0f
                                        };
    const std::vector<float> inputs_bin_conv{
                                        // depth: 1
                                        1.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 1.0f, 0.0f, 1.0f,
                                        1.0f, 1.0f, 0.0f, 1.0f,
                                        1.0f, 0.0f, 1.0f, 0.0f,
                                        // depth: 2
                                        0.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 1.0f, 0.0f, 1.0f,
                                        1.0f, 0.0f, 1.0f, 0.0f,
                                        // depth: 3
                                        1.0f, 1.0f, 1.0f, 0.0f,
                                        0.0f, 1.0f, 0.0f, 1.0f,
                                        1.0f, 1.0f, 0.0f, 1.0f,
                                        1.0f, 0.0f, 1.0f, 0.0f,
                                        // depth: 4
                                        1.0f, 0.0f, 1.0f, 0.0f,
                                        1.0f, 1.0f, 0.0f, 1.0f,
                                        0.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 0.0f, 1.0f
                                        };

    const Shape filters_shape{1, 1, 3, 3, 3};
    const std::vector<float> filters_conv{
                                         // depth: 1
                                         1.0f, -1.0f, 1.0f,
                                         -1.0f, 1.0f, -1.0f,
                                         1.0f, -1.0f, 1.0f,
                                         // depth: 2
                                         -1.0f, 1.0f, 1.0f,
                                         1.0f, -1.0f, 1.0f,
                                         1.0f, 1.0f, -1.0f,
                                         // depth: 3
                                         1.0f, 1.0f, -1.0f,
                                         -1.0f, 1.0f, -1.0f,
                                         1.0f, 1.0f, 1.0f};
     const std::vector<uint8_t> filters_bin_conv{0xAA, 0xBB, 0xB2, 0xE0};

    const Shape outputs_shape{1, 1, 2, 2, 2};
    const std::vector<float> outputs{
                                     // depth: 1
                                     13.0f, 3.0f,
                                     -3.0f, -3.0f,
                                     // depth: 2
                                     -3.0f, 5.0f,
                                     11.0f, -3.0f};

    BinaryConvolutionTest(
        inputs_bin_conv,
        inputs_shape,
        filters_bin_conv,
        filters_shape,
        outputs,
        outputs_shape,
        strides,
        padding,
        dilations);

    ConvolutionTest(
        inputs_conv,
        inputs_shape,
        filters_conv,
        filters_shape,
        outputs,
        outputs_shape,
        strides,
        padding,
        dilations);
}
// clang-format off
