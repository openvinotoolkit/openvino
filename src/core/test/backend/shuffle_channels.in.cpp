// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});


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
    // In this test the output is the same as in shuffle_channels_simple but
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

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_1d)
{
    Shape data_shape{15};
    const auto data = make_shared<op::Parameter>(element::i32, data_shape);
    auto tested_op = make_shared<op::ShuffleChannels>(data, 0, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    std::vector<int32_t> input_data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14};

    test_case.add_input(input_data);
    test_case.add_expected_output<int32_t>(
        data_shape,
        {0, 3, 6, 9, 12,
        1, 4, 7, 10, 13,
        2, 5, 8, 11, 14});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_2d)
{
    Shape data_shape{15, 4};
    const auto data = make_shared<op::Parameter>(element::i32, data_shape);
    auto tested_op = make_shared<op::ShuffleChannels>(data, 0, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    std::vector<int32_t> input_data{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};

    test_case.add_input(input_data);
    test_case.add_expected_output<int32_t>(
        data_shape,
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_3d)
{
    Shape data_shape{15, 2, 2};
    const auto data = make_shared<op::Parameter>(element::i32, data_shape);
    auto tested_op = make_shared<op::ShuffleChannels>(data, 0, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);


    std::vector<int32_t> input_data{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};

    test_case.add_input(input_data);
    test_case.add_expected_output<int32_t>(
        data_shape,
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_5d)
{
    Shape data_shape{2, 2, 15, 2, 2};
    const auto data = make_shared<op::Parameter>(element::i32, data_shape);
    auto tested_op = make_shared<op::ShuffleChannels>(data, 2, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(function);

    std::vector<int32_t> input_data{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};

    test_case.add_input(input_data);
    test_case.add_expected_output<int32_t>(
        data_shape,
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59,

         0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59,

         0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59,

         0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}
