//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "op/group_conv.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, elu)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::Elu>(A, 0.5f);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::Elu>(A, -1.f);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu)
{
    Shape shape{3, 2};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{0, 0.5, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{0, 3, -1, 1, -1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, hardsigmoid)
{
    const Shape shape{2, 7};
    const float alpha_f = 0.125f;
    const float beta_f = 0.642f;
    const auto A = make_shared<op::Parameter>(element::f32, shape);
    const auto alpha = op::Constant::create<float>(A->get_element_type(), Shape{}, {alpha_f});
    const auto beta = op::Constant::create<float>(A->get_element_type(), Shape{}, {beta_f});
    auto hardsigmoid = make_shared<op::HardSigmoid>(A, alpha, beta);
    auto f = make_shared<Function>(NodeVector{hardsigmoid}, ParameterVector{A});
    vector<float> input{-1.f,
                        0.f,
                        1.f,
                        -100.f,
                        100.f,
                        -3.1234567f,
                        5.876543f,
                        7.13245364f,
                        numeric_limits<float>::max(),
                        numeric_limits<float>::lowest(),
                        numeric_limits<float>::min(),
                        numeric_limits<float>::infinity(),
                        numeric_limits<float>::min() / 16.f,
                        -numeric_limits<float>::min() / 16.f};

    // Prepare expected output data
    auto impl = [alpha_f, beta_f](float val) { return min(max(alpha_f * val + beta_f, 0.f), 1.f); };
    vector<float> expected_output;
    transform(begin(input), end(input), back_inserter(expected_output), impl);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(input);
    test_case.add_expected_output<float>(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_shared_slope)
{
    Shape shape{3, 2};
    Shape rshape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{0.5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{-1, 3, -1, 1, -0.5, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_slope)
{
    Shape shape{3, 2};
    Shape rshape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{-0.5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{1, 3, 1, 1, 0.5, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2},
                                         vector<float>{11, 14, 17, 20, 79, 86, 93, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_striding)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{2, 2},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(Shape{1, 2, 1, 1}, vector<float>{11, 79});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_window_dilation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2},
                                         vector<float>{11, 14, 17, 20, 79, 86, 93, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_data_dilation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{2, 2},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(
        Shape{1, 2, 3, 3},
        vector<float>{11, 0, 14, 0, 0, 0, 17, 0, 20, 79, 0, 86, 0, 0, 0, 93, 0, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_padding)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(
        Shape{1, 2, 3, 3},
        vector<float>{0, 0, 0, 11, 14, 0, 17, 20, 0, 0, 0, 0, 79, 86, 0, 93, 100, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_padding_and_window_dilation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(
        Shape{1, 2, 3, 3},
        vector<float>{0, 0, 0, 11, 14, 0, 17, 20, 0, 0, 0, 0, 79, 86, 0, 93, 100, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_input_shape_variation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 4, 1});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(
        Shape{1, 2, 5, 2},
        vector<float>{0, 0, 11, 0, 14, 0, 17, 0, 20, 0, 0, 0, 79, 0, 86, 0, 93, 0, 100, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_input_data_variation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 3, 3});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4},
        vector<float>{0, 0, 0, 0, 21,  24,  27,  0, 30,  33,  36,  0, 39,  42,  45,  0,
                      0, 0, 0, 0, 169, 176, 183, 0, 190, 197, 204, 0, 211, 218, 225, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_groups_included_in_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
    auto f = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2},
                                         vector<float>{11, 14, 17, 20, 79, 86, 93, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_depth_block_first)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4, 4});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);
    auto function = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                                11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                                22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.add_expected_output<float>(Shape{1, 8, 2, 2},
                                         {
                                             0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f,
                                             1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
                                             4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f,
                                             5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
                                         });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_depth_depth_first)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4, 4});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);
    auto function = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.f,  16.f, 2.f,  18.f, 1.f,  17.f, 3.f,  19.f, 8.f,  24.f, 10.f,
                                26.f, 9.f,  25.f, 11.f, 27.f, 4.f,  20.f, 6.f,  22.f, 5.f,  21.f,
                                7.f,  23.f, 12.f, 28.f, 14.f, 30.f, 13.f, 29.f, 15.f, 31.f});
    test_case.add_expected_output<float>(
        Shape{1, 8, 2, 2}, {0.f,  2.f,  8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f,  3.f,  9.f,
                            11.f, 17.f, 19.f, 25.f, 27.f, 4.f,  6.f,  12.f, 14.f, 20.f, 22.f,
                            28.f, 30.f, 5.f,  7.f,  13.f, 15.f, 21.f, 23.f, 29.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_block_first)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
    auto function = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    });
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                            11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                            22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_depth_first)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    auto function = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    });
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  16.f, 2.f,  18.f, 1.f,  17.f, 3.f,  19.f, 8.f,  24.f, 10.f,
                            26.f, 9.f,  25.f, 11.f, 27.f, 4.f,  20.f, 6.f,  22.f, 5.f,  21.f,
                            7.f,  23.f, 12.f, 28.f, 14.f, 30.f, 13.f, 29.f, 15.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01428571f, 0.02857143f, 0.04285714f, 0.05714286f, 0.07142857f, 0.08571429f,
                     0.1f,        0.11428571f, 0.12857144f, 0.14285715f, 0.15714286f, 0.17142858f,
                     0.18571429f, 0.2f,        0.21428572f, 0.22857143f, 0.24285714f, 0.25714287f,
                     0.27142859f, 0.2857143f,  0.30000001f, 0.31428573f, 0.32857144f, 0.34285715f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_empty_axes_input)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    // output should be filled with 1f values
    test_case.add_expected_output<float>(data_shape, vector<float>(shape_size(data_shape), 1));

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_h_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_1axis_5d)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_123axes_5d)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.02638899f, 0.04956816f, 0.070014f,   0.10555596f, 0.1239204f,  0.140028f,
                     0.18472293f, 0.19827265f, 0.210042f,   0.26388991f, 0.27262488f, 0.280056f,
                     0.34305686f, 0.34697714f, 0.35007f,    0.42222384f, 0.42132938f, 0.420084f,
                     0.50139081f, 0.49568161f, 0.49009803f, 0.58055776f, 0.57003385f, 0.560112f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_c_2x2_shape)
{
    Shape data_shape{2, 2};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.44721353f, 0.89442706f, 0.60000002f, 0.80000001f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_c_2x4_shape)
{
    Shape data_shape{2, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.18257418f,
                                          0.36514837f,
                                          0.54772252f,
                                          0.73029673f,
                                          0.37904903f,
                                          0.45485884f,
                                          0.53066862f,
                                          0.60647845f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_4d_max_bias)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{5000};
    auto eps_mode = op::EpsMode::MAX;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01414214f, 0.02828427f, 0.04242641f, 0.05656854f, 0.07071068f, 0.08485281f,
                     0.09899495f, 0.11313709f, 0.12727922f, 0.14142136f, 0.15556349f, 0.16970563f,
                     0.18384777f, 0.1979899f,  0.21213204f, 0.22627418f, 0.2404163f,  0.25455844f,
                     0.26870057f, 0.28284273f, 0.29698485f, 0.31112698f, 0.32526913f, 0.33941126f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

namespace
{
    template <typename T, test::TestCaseType tct = test::TestCaseType::STATIC>
    void clamp_test(const element::Type& type,
                    const PartialShape& dynamic_shape,
                    const Shape& static_shape,
                    const std::vector<T>& input,
                    double min,
                    double max,
                    const std::vector<T>& output)
    {
        auto data = make_shared<op::Parameter>(type, dynamic_shape);
        auto clamp = make_shared<op::Clamp>(data, min, max);
        auto function = make_shared<Function>(clamp, ParameterVector{data});

        auto test_case = test::TestCase<TestEngine, tct>(function);
        test_case.template add_input<T>(static_shape, input);
        test_case.template add_expected_output<T>(static_shape, output);
        return test_case.run();
    }
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_double)
{
    auto type = element::f64;
    typedef double ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
        0.2,
        0.6,
        {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        20.0,
        {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_float)
{
    auto type = element::f32;
    typedef float ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
        0.2,
        0.6,
        {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        20.0,
        {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_int8)
{
    auto type = element::i8;
    typedef int8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_int16)
{
    auto type = element::i16;
    typedef int16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_int32)
{
    auto type = element::i32;
    typedef int32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_int64)
{
    auto type = element::i64;
    typedef int64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_uint8)
{
    auto type = element::u8;
    typedef uint8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_uint16)
{
    auto type = element::u16;
    typedef uint16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_uint32)
{
    auto type = element::u32;
    typedef uint32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (numeric_limits<ctype>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_uint64)
{
    auto type = element::u64;
    typedef uint64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    // TODO: Fix CPU DEX / MLIR correctness bug: using signed comparison for unsigned ints
    // auto max = numeric_limits<ctype>::max();
    // auto pinf = numeric_limits<double>::infinity();
    ctype max = (static_cast<ctype>(1) << (32 - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype>(type, sshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype>(type, sshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, 20.0, {10, 20, 10, 10, 11, 19, 20, 20});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, 10.0, pinf, {10, max, 10, 10, 11, 19, 20, 21});
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type, dshape, sshape, input, ninf, 20.0, {min, 20, 9, 10, 11, 19, 20, 20});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_float16)
{
    auto type = element::f16;
    typedef float16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
        0.2,
        0.6,
        {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        20.0,
        {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp_bfloat16)
{
    auto type = element::bf16;
    typedef bfloat16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                      0.2,
                      0.6,
                      {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      20.0,
                      {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      10.0,
                      pinf,
                      {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype>(type,
                      sshape,
                      sshape,
                      input,
                      ninf,
                      20.0,
                      {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    // dynamic shape
    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
        0.2,
        0.6,
        {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        20.0,
        {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001});

    clamp_test<ctype, test::TestCaseType::DYNAMIC>(
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0});
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, true, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(
        data_shape, vector<float>{-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization_split_channels)
{
    Shape data_shape{1, 2, 5, 1};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>({1, 2, 5, 1},
                                         vector<float>{-2, -1, 0, 1, 2, -2, -1, 0, 1, 2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.566698903055826,
                                                       -1.2185435912656424,
                                                       -0.87038827947545883,
                                                       -0.52223296768527527,
                                                       -0.17407765589509178,
                                                       0.17407765589509178,
                                                       0.52223296768527527,
                                                       0.87038827947545883,
                                                       1.2185435912656424,
                                                       1.566698903055826});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization_split_channels)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948,
                                                       -1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, grn_4d)
{
    const Shape data_shape{1, 2, 3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    float bias{1e-6f};

    const auto grn = make_shared<op::GRN>(data, bias);
    const auto function = make_shared<Function>(NodeVector{grn}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,
                     0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f, 0.9593655f,  0.9486833f,
                     0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, grn_2d_with_bias)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    float bias{2.25f};

    const auto grn = make_shared<op::GRN>(data, bias);
    const auto function = make_shared<Function>(NodeVector{grn}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.5547002f,
                                          0.8f,
                                          0.8944272f,
                                          0.9363292f,
                                          0.95782626f,
                                          0.9701425f,
                                          0.9778024f,
                                          0.98287225f,
                                          0.9863939f,
                                          0.9889363f,
                                          0.9908301f,
                                          0.99227786f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, unsqueeze)
{
    auto data_node = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto squeeze = make_shared<op::Unsqueeze>(data_node, axes_node);

    auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 1, 2}, data);
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_simple)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 15, 2, 2});
    auto tested_op = make_shared<op::ShuffleChannels>(data, 1, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    std::vector<int32_t> input_data(60);
    std::iota(std::begin(input_data), std::end(input_data), 0);
    test_case.add_input(input_data);

    test_case.add_expected_output<int32_t>(
        Shape{1, 15, 2, 2},
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_negative_axis)
{
    // in this test the output is the same as in shuffle_channels_simple but
    // the axis value is negative and the C(channels) value is in a different dimension(0) of the
    // shape
    const auto data = make_shared<op::Parameter>(element::i32, Shape{15, 2, 1, 2});
    auto tested_op = make_shared<op::ShuffleChannels>(data, -4, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    std::vector<int32_t> input_data(60);
    std::iota(std::begin(input_data), std::end(input_data), 0);
    test_case.add_input(input_data);

    test_case.add_expected_output<int32_t>(
        Shape{15, 2, 1, 2},
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_float)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{6, 1, 1, 1});
    auto tested_op = make_shared<op::ShuffleChannels>(data, 0, 2);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    test_case.add_expected_output<float>(Shape{6, 1, 1, 1}, {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    const auto squeeze = make_shared<op::Squeeze>(data_node, axes_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze_default_axes)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    const auto squeeze = make_shared<op::Squeeze>(data_node, axes_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze_dynamic)
{
    const auto data_param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_param = make_shared<op::Parameter>(element::i64, Shape{2});
    EXPECT_THROW(make_shared<op::Squeeze>(data_param, axes_param), CheckFailure);
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference)
{
    const auto x1 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto tested_op = make_shared<op::SquaredDifference>(x1, x2);
    auto function = make_shared<Function>(tested_op, ParameterVector{x1, x2});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.0, 16.0, 0.0, 1.234567});
    test_case.add_input<float>({1.0, 8.0, -3.0, 3.456789});

    test_case.add_expected_output<float>(Shape{2, 2}, {0.0, 64.0, 9.0, 4.938270617284});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_broadcast)
{
    const auto x1 = make_shared<op::Parameter>(element::i32, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::i32, Shape{});

    auto tested_op = make_shared<op::SquaredDifference>(x1, x2);
    auto function = make_shared<Function>(tested_op, ParameterVector{x1, x2});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int32_t>({1, 1, 1, 1});
    test_case.add_input<int32_t>({1});

    test_case.add_expected_output<int32_t>(Shape{2, 2}, {0, 0, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_3_equal_parts)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{6});
    const auto axis = op::Constant::create(element::i64, Shape{}, {0});

    const auto tested_op = make_shared<op::Split>(data, axis, 3);
    const auto function = make_shared<Function>(tested_op->decompose_op(), ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6});

    test_case.add_expected_output<int32_t>(Shape{2}, {1, 2});
    test_case.add_expected_output<int32_t>(Shape{2}, {3, 4});
    test_case.add_expected_output<int32_t>(Shape{2}, {5, 6});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_var_len_parts)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});

    const std::vector<size_t> splits = {2, 4};
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto tested_op = make_shared<op::Split>(data, axis, splits);
    const auto function = make_shared<Function>(tested_op->decompose_op(), ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<int32_t>(Shape{2, 2}, {0, 1, 6, 7});
    test_case.add_expected_output<int32_t>(Shape{2, 4}, {2, 3, 4, 5, 8, 9, 10, 11});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_zero_bias_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(
        X, H_t, C_t, W, R, B, P, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B(gates_count * hidden_size, 0.f);
    // P
    vector<float> in_P(3 * hidden_size, 0.f);

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(
        X, H_t, C_t, W, R, B, P, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9218244f, 0.78787273f, 0.8754273f, 0.7361462f, 0.70927656f, 0.83522964f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.7094649f, 1.1259761f, 1.444019f, 1.086587f, 0.9762144f, 1.3066899f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_peepholes_clip_input_forget)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;
    bool input_forget = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X,
                                                     H_t,
                                                     C_t,
                                                     W,
                                                     R,
                                                     B,
                                                     P,
                                                     hidden_size,
                                                     op::LSTMWeightsFormat::IOFC,
                                                     vector<string>{"sigmoid", "tanh", "tanh"},
                                                     vector<float>{},
                                                     vector<float>{},
                                                     clip_threshold,
                                                     input_forget);
    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.71485436f, 0.71844107f, 0.72704613f, 0.6235602f, 0.68306124f, 0.6978715f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94656503f, 0.9527454f, 0.9706756f, 0.84206575f, 0.91898793f, 0.9127192f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_activaction_functions)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;
    bool input_forget = true;
    vector<string> activations{"sigmoid", "tanh", "hardsigmoid"};
    vector<float> activation_alpha{0.f, 0.f, 1.8345f};
    vector<float> activation_beta{0.f, 0.f, 3.05f};

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X,
                                                     H_t,
                                                     C_t,
                                                     W,
                                                     R,
                                                     B,
                                                     P,
                                                     hidden_size,
                                                     op::LSTMWeightsFormat::IOFC,
                                                     activations,
                                                     activation_alpha,
                                                     activation_beta,
                                                     clip_threshold,
                                                     input_forget);
    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.96834344f, 0.9695254f, 0.97068775f, 0.9077866f, 0.94161016f, 0.96599925f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94656503f, 0.9527454f, 0.9706756f, 0.84206575f, 0.91898793f, 0.9127192f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 4;
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    const auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({0.0f});
    // input_high
    test_case.add_input<float>({23.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,          2.f,          2.f,          2.f,          6.6666669f,
                      6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                      6.6666669f,   6.6666669f,   11.33333301f, 11.33333301f, 11.33333301f,
                      11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                      16.f,         16.f,         16.f,         16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 5;
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    const auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({3.f});
    // input_high
    test_case.add_input<float>({17.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                      12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip_across_channels)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{2, 1, 1});

    auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_pdpd)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{2});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{2});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{2});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{2});

    auto quantize =
        make_shared<op::FakeQuantize>(data,
                                      input_low,
                                      input_high,
                                      output_low,
                                      output_high,
                                      levels,
                                      op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));
    auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = test::TestCase<TestEngine>(function);

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_no_bias)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X, H_t, W, R, hidden_size);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9408395f, 0.53823817f, 0.84270686f, 0.98932856f, 0.768665f, 0.90461975f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X,
                                                   H_t,
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size,
                                                   vector<string>{"tanh"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({1.0289404f, 1.6362579f, 0.4370661f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9922437f, 0.97749525f, 0.9312212f, 0.9937176f, 0.9901317f, 0.95906746f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X,
                                                   H_t,
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size,
                                                   vector<string>{"sigmoid"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({1.0289404f, 1.6362579f, 0.4370661f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94126844f, 0.9036043f, 0.841243f, 0.9468489f, 0.934215f, 0.873708f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = false;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X,
                                                   H_t,
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size,
                                                   vector<string>{"sigmoid", "tanh"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip,
                                                   linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f});

    // Ht
    test_case.add_input<float>(
        {0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f});

    // W
    test_case.add_input<float>(
        {0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
         0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
         0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
         0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f});

    // R
    test_case.add_input<float>(
        {0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f, 0.1623331f,  0.01389796f, 0.99669236f,
         0.53901845f, 0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f, 0.40111724f, 0.37497616f,
         0.2903471f,  0.6796794f,  0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f, 0.791677f,
         0.76497287f, 0.9895242f,  0.7845312f,  0.51267904f, 0.49030215f, 0.08498167f});

    // B (the sum of biases for W and R)
    test_case.add_input<float>({
        0.8286678f + 0.9175602f,
        0.9153158f + 0.14958014f,
        0.9581612f + 0.49230585f,
        0.6639213f + 0.63162816f,
        0.84239805f + 0.4161903f,
        0.5282445f + 0.22148274f,
        0.14153397f + 0.50496656f,
        0.22404431f + 0.34798595f,
        0.6549655f + 0.6699164f,
    });

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.48588726f, 0.99670005f, 0.83759373f, 0.5023099f, 0.89410484f, 0.60011315f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_linear_before_reset)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{(gates_count + 1) * hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X,
                                                   H_t,
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size,
                                                   vector<string>{"sigmoid", "tanh"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip,
                                                   linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});
    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});

    // B (the sum of biases for W and R for z and r gates, and separately for W and R for h gate)
    test_case.add_input<float>({0.61395123f, // 0.09875853f + 0.5151927f,
                                1.08667738f, // 0.37801138f + 0.708666f,
                                1.32600244f, // 0.7729636f + 0.55303884f,
                                0.81917698f, // 0.78493553f + 0.03424145f,
                                1.37736335f, // 0.5662702f + 0.81109315f,
                                0.42931147f, // 0.12406381f + 0.30524766f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8709214f, 0.48411977f, 0.74495184f, 0.6074972f, 0.44572943f, 0.1467715f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{(gates_count + 1) * hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X,
                                                   H_t,
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size,
                                                   vector<string>{"hardsigmoid", "hardsigmoid"},
                                                   vector<float>{1.8345f, 1.8345f},
                                                   vector<float>{3.05f, 3.05f},
                                                   clip,
                                                   linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});

    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});

    // B (the sum of biases for W and R for z and r gates, and separately for W and R for h gate)
    test_case.add_input<float>({0.09875853f + 0.5151927f,
                                0.37801138f + 0.708666f,
                                0.7729636f + 0.55303884f,
                                0.78493553f + 0.03424145f,
                                0.5662702f + 0.81109315f,
                                0.12406381f + 0.30524766f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_space_to_depth_block_first)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape dts_input_shape{2, 32, 2, 4, 2, 4};
    size_t block_size = 2;

    auto dts_input = make_shared<op::Parameter>(element::f32, dts_input_shape);
    auto depth_to_space = make_shared<op::DepthToSpace>(
        dts_input, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, block_size);
    auto dts_func = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{dts_input});

    auto dts_input_tensor = backend->create_tensor(element::f32, dts_input_shape);
    const auto data_size = shape_size(dts_input_shape);
    vector<float> data(data_size);
    std::iota(data.begin(), data.end(), 0);
    copy_data(dts_input_tensor, data);
    const auto dts_output_shape = depth_to_space->get_output_shape(0);
    auto dts_output_tensor = backend->create_tensor(element::f32, dts_output_shape);
    auto handle = backend->compile(dts_func);
    handle->call_with_validate({dts_output_tensor}, {dts_input_tensor});
    auto dts_result = read_vector<float>(dts_output_tensor);

    // use depth_to_space output as space_to_depth input
    auto std_input = make_shared<op::Parameter>(element::f32, dts_output_shape);
    auto space_to_depth = make_shared<op::SpaceToDepth>(
        std_input, op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size);
    auto std_func = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{std_input});

    auto std_input_tensor = backend->create_tensor(element::f32, dts_output_shape);
    copy_data(std_input_tensor, dts_result);
    auto std_output_tensor = backend->create_tensor(element::f32, dts_input_shape);
    handle = backend->compile(std_func);
    handle->call_with_validate({std_output_tensor}, {std_input_tensor});
    auto std_result = read_vector<float>(std_output_tensor);

    // expected output of space_to_depth is input of depth_to_space
    ASSERT_EQ(dts_input_shape, space_to_depth->get_output_shape(0));
    EXPECT_TRUE(test::all_close_f(std_result, data, data_size));
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_space_to_depth_depth_first)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape dts_input_shape{2, 32, 2, 4, 2, 4};
    size_t block_size = 2;

    auto dts_input = make_shared<op::Parameter>(element::f32, dts_input_shape);
    auto depth_to_space = make_shared<op::DepthToSpace>(
        dts_input, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, block_size);
    auto dts_func = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{dts_input});

    auto dts_input_tensor = backend->create_tensor(element::f32, dts_input_shape);
    const auto data_size = shape_size(dts_input_shape);
    vector<float> data(data_size);
    std::iota(data.begin(), data.end(), 0);
    copy_data(dts_input_tensor, data);
    const auto dts_output_shape = depth_to_space->get_output_shape(0);
    auto dts_output_tensor = backend->create_tensor(element::f32, dts_output_shape);
    auto handle = backend->compile(dts_func);
    handle->call_with_validate({dts_output_tensor}, {dts_input_tensor});
    auto dts_result = read_vector<float>(dts_output_tensor);

    // use depth_to_space output as space_to_depth input
    auto std_input = make_shared<op::Parameter>(element::f32, dts_output_shape);
    auto space_to_depth = make_shared<op::SpaceToDepth>(
        std_input, op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size);
    auto std_func = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{std_input});

    auto std_input_tensor = backend->create_tensor(element::f32, dts_output_shape);
    copy_data(std_input_tensor, dts_result);
    auto std_output_tensor = backend->create_tensor(element::f32, dts_input_shape);
    handle = backend->compile(std_func);
    handle->call_with_validate({std_output_tensor}, {std_input_tensor});
    auto std_result = read_vector<float>(std_output_tensor);

    // expected output of space_to_depth is input of depth_to_space
    ASSERT_EQ(dts_input_shape, space_to_depth->get_output_shape(0));
    EXPECT_TRUE(test::all_close_f(std_result, data, data_size));
}
