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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, split_1d)
{
    const auto data = make_shared<op::Parameter>(element::Type_t::i32, Shape{6});
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {0});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 3);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6});

    test_case.add_expected_output<int32_t>(Shape{2}, {1, 2});
    test_case.add_expected_output<int32_t>(Shape{2}, {3, 4});
    test_case.add_expected_output<int32_t>(Shape{2}, {5, 6});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_2d_axis_0)
{
    Shape shape{6, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {0});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 2);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{3, 2}, {0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>(Shape{3, 2}, {6, 7, 8, 9, 10, 11});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_2d_axis_1)
{
    Shape shape{6, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {1});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 2);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{6, 1}, {0, 2, 4, 6, 8, 10});
    test_case.add_expected_output<float>(Shape{6, 1}, {1, 3, 5, 7, 9, 11});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_3d_axis_0)
{
    Shape shape{2, 2, 3};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {0});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 2);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{1, 2, 3}, {0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>(Shape{1, 2, 3}, {6, 7, 8, 9, 10, 11});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_3d_axis_1)
{
    Shape shape{2, 8, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {1});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 4);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{2, 2, 2}, {0, 1, 2, 3, 16, 17, 18, 19});
    test_case.add_expected_output<float>(Shape{2, 2, 2}, {4, 5, 6, 7, 20, 21, 22, 23});
    test_case.add_expected_output<float>(Shape{2, 2, 2}, {8, 9, 10, 11, 24, 25, 26, 27});
    test_case.add_expected_output<float>(Shape{2, 2, 2}, {12, 13, 14, 15, 28, 29, 30, 31});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_3d_axis_2)
{
    Shape shape{2, 1, 6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {2});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 2);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{2, 1, 3}, {0, 1, 2, 6, 7, 8});
    test_case.add_expected_output<float>(Shape{2, 1, 3}, {3, 4, 5, 9, 10, 11});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_4d_axis_0)
{
    Shape shape{3, 2, 3, 1};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {0});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 3);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{1, 2, 3, 1}, {0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>(Shape{1, 2, 3, 1}, {6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>(Shape{1, 2, 3, 1}, {12, 13, 14, 15, 16, 17});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_4d_axis_1)
{
    Shape shape{2, 8, 2, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {1});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 4);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{2, 2, 2, 2},
                                         {0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39});
    test_case.add_expected_output<float>(
        Shape{2, 2, 2, 2}, {8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47});
    test_case.add_expected_output<float>(
        Shape{2, 2, 2, 2}, {16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55});
    test_case.add_expected_output<float>(
        Shape{2, 2, 2, 2}, {24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_4d_axis_2)
{
    Shape shape{2, 1, 6, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {2});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 3);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{2, 1, 2, 2}, {0, 1, 2, 3, 12, 13, 14, 15});
    test_case.add_expected_output<float>(Shape{2, 1, 2, 2}, {4, 5, 6, 7, 16, 17, 18, 19});
    test_case.add_expected_output<float>(Shape{2, 1, 2, 2}, {8, 9, 10, 11, 20, 21, 22, 23});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_4d_axis_3)
{
    Shape shape{2, 1, 2, 6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, shape);
    const auto axis = op::Constant::create(element::Type_t::i64, Shape{}, {3});

    const auto tested_op = make_shared<op::v1::Split>(data, axis, 3);
    const auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> in(shape_size(shape));
    std::iota(in.begin(), in.end(), 0);
    test_case.add_input<float>(in);

    test_case.add_expected_output<float>(Shape{2, 1, 2, 2}, {0, 1, 6, 7, 12, 13, 18, 19});
    test_case.add_expected_output<float>(Shape{2, 1, 2, 2}, {2, 3, 8, 9, 14, 15, 20, 21});
    test_case.add_expected_output<float>(Shape{2, 1, 2, 2}, {4, 5, 10, 11, 16, 17, 22, 23});

    test_case.run();
}
