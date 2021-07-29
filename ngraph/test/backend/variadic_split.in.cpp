// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

namespace
{
    template <typename T>
    std::shared_ptr<ngraph::Function> StaticVariadicSplit(ngraph::element::Type_t inputs_type,
                                                          Shape data_shape,
                                                          Shape axis_shape,
                                                          Shape split_lenghts_shape,
                                                          std::vector<T> axis_value,
                                                          std::vector<T> split_lenghts_value)
    {
        const auto data = make_shared<op::Parameter>(inputs_type, data_shape);
        const auto axis = op::Constant::create(inputs_type, axis_shape, axis_value);
        const auto split_lengths =
            op::Constant::create(inputs_type, split_lenghts_shape, split_lenghts_value);
        const auto variadic_split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);
        return make_shared<Function>(variadic_split, ParameterVector{data});
    }

    std::shared_ptr<ngraph::Function> DynamicVariadicSplit(ngraph::element::Type_t inputs_type,
                                                           Shape data_shape,
                                                           Shape axis_shape,
                                                           Shape split_lenghts_shape)
    {
        const auto data = make_shared<op::Parameter>(inputs_type, data_shape);
        const auto axis = make_shared<op::Parameter>(inputs_type, axis_shape);
        const auto split_lengths = make_shared<op::Parameter>(inputs_type, split_lenghts_shape);
        const auto variadic_split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);
        return make_shared<Function>(variadic_split, ParameterVector{data, axis, split_lengths});
    }
} // namespace

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_1d_static)
{
    const Shape data_shape{10};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{5, 3, 2};

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_expected_output<int32_t>(Shape{3}, {6, 7, 8});
    test_case.add_expected_output<int32_t>(Shape{2}, {9, 10});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_1d_dynamic)
{
    const Shape data_shape{10};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{5, 3, 2};

    const auto f = DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape);
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_expected_output<int32_t>(Shape{3}, {6, 7, 8});
    test_case.add_expected_output<int32_t>(Shape{2}, {9, 10});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_2d_axis_0_static)
{
    const Shape data_shape{6, 2};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Shape axis_shape{}; // scalar
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{2};
    const std::vector<int32_t> split_lenghts{4, 2};

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {9, 10, 11, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_2d_axis_0_dynamic)
{
    const Shape data_shape{6, 2};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Shape axis_shape{}; // scalar
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{2};
    const std::vector<int32_t> split_lenghts{4, 2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(
        DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape));

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(Shape{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {9, 10, 11, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_2d_axis_1_static)
{
    const Shape data_shape{4, 3};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Shape axis_shape{}; // scalar
    const std::vector<int32_t> axis_value{1};
    const Shape split_lenghts_shape{2};
    const std::vector<int32_t> split_lenghts{1, 2};

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{4, 1}, {1, 4, 7, 10});
    test_case.add_expected_output<int32_t>(Shape{4, 2}, {2, 3, 5, 6, 8, 9, 11, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_2d_axis_1_dynamic)
{
    const Shape data_shape{4, 3};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Shape axis_shape{}; // scalar
    const std::vector<int32_t> axis_value{1};
    const Shape split_lenghts_shape{2};
    const std::vector<int32_t> split_lenghts{1, 2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(
        DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape));

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(Shape{4, 1}, {1, 4, 7, 10});
    test_case.add_expected_output<int32_t>(Shape{4, 2}, {2, 3, 5, 6, 8, 9, 11, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_0_static)
{
    const Shape data_shape{6, 2, 3, 1};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);
    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{3, 1, 2};

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(
        Shape{3, 2, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
    test_case.add_expected_output<int32_t>(Shape{1, 2, 3, 1}, {18, 19, 20, 21, 22, 23});
    test_case.add_expected_output<int32_t>(Shape{2, 2, 3, 1},
                                           {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_0_dynamic)
{
    const Shape data_shape{6, 2, 3, 1};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);
    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{3, 1, 2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(
        DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape));

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(
        Shape{3, 2, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
    test_case.add_expected_output<int32_t>(Shape{1, 2, 3, 1}, {18, 19, 20, 21, 22, 23});
    test_case.add_expected_output<int32_t>(Shape{2, 2, 3, 1},
                                           {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_1_static)
{
    const Shape data_shape{2, 8, 2, 2};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);

    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{1};
    const Shape split_lenghts_shape{4};
    const std::vector<int32_t> split_lenghts{1, 3, 2, 2};

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 2}, {0, 1, 2, 3, 32, 33, 34, 35});
    test_case.add_expected_output<int32_t>(
        Shape{2, 3, 2, 2},
        {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
    test_case.add_expected_output<int32_t>(
        Shape{2, 2, 2, 2}, {16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55});
    test_case.add_expected_output<int32_t>(
        Shape{2, 2, 2, 2}, {24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_1_dynamic)
{
    const Shape data_shape{2, 8, 2, 2};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);

    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{1};
    const Shape split_lenghts_shape{4};
    const std::vector<int32_t> split_lenghts{1, 3, 2, 2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(
        DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape));

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 2}, {0, 1, 2, 3, 32, 33, 34, 35});
    test_case.add_expected_output<int32_t>(
        Shape{2, 3, 2, 2},
        {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
    test_case.add_expected_output<int32_t>(
        Shape{2, 2, 2, 2}, {16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55});
    test_case.add_expected_output<int32_t>(
        Shape{2, 2, 2, 2}, {24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_2_static)
{
    const Shape data_shape{2, 1, 6, 2};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);

    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{2};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{3, 1, 2};

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{2, 1, 3, 2},
                                           {0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 1, 2}, {6, 7, 18, 19});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 2}, {8, 9, 10, 11, 20, 21, 22, 23});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_2_dynamic)
{
    const Shape data_shape{2, 1, 6, 2};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);

    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{2};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{-1, 1, 2}; // -1 means "all remaining items"

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(
        DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape));

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(Shape{2, 1, 3, 2},
                                           {0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 1, 2}, {6, 7, 18, 19});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 2}, {8, 9, 10, 11, 20, 21, 22, 23});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_3_static)
{
    const Shape data_shape{2, 1, 2, 6};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);

    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{3};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{1, -1, 3}; // -1 means "all remaining items"

    auto test_case = test::TestCase<TestEngine>(StaticVariadicSplit(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 1}, {0, 6, 12, 18});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 2}, {1, 2, 7, 8, 13, 14, 19, 20});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 3},
                                           {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_4d_axis_3_dynamic)
{
    const Shape data_shape{2, 1, 2, 6};
    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 0);

    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{3};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{1, 2, -1}; // -1 means "all remaining items"

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(
        DynamicVariadicSplit(element::i32, data_shape, axis_shape, split_lenghts_shape));

    test_case.add_input(data_shape, data);
    test_case.add_input(axis_shape, axis_value);
    test_case.add_input(split_lenghts_shape, split_lenghts);
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 1}, {0, 6, 12, 18});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 2}, {1, 2, 7, 8, 13, 14, 19, 20});
    test_case.add_expected_output<int32_t>(Shape{2, 1, 2, 3},
                                           {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23});

    test_case.run();
}