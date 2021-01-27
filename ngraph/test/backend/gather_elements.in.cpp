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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, evaluate_1D_gather_elements_3_indices_int32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{7});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    test_case.add_input<int32_t>({1, 2, 3});
    test_case.add_input<int32_t>({1, 2, 0, 2, 0, 0, 2});
    test_case.add_expected_output<int32_t>(vector<int32_t>{2, 3, 1, 3, 1, 1, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x2_indices_int32_axis_0)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<int32_t>({1, 2,
                                  3, 4});
    test_case.add_input<int32_t>({0, 1,
                                  0, 0});

    test_case.add_expected_output<int32_t>(vector<int32_t>{1, 4,
                                                           1, 2});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x2_indices_int32_axis_1)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = 1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{1, 2,
                              3, 4};
    std::vector<int32_t> indices{0, 1,
                                 0, 0};

    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{1, 2,
                                                           3, 3});
    test_case.run();
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x2_indices_int32_axis_minus_1)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = -1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{1, 2,
                              3, 4};
    std::vector<int32_t> indices{0, 1,
                                 0, 0};

    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{1, 2,
                                                           3, 3});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x3_indices_int32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{1, 2, 3,
                              4, 5, 6,
                              7, 8, 9};
    std::vector<int32_t> indices{1, 2, 0,
                                 2, 0, 0};

    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{4, 8, 3,
                                                           7, 2, 3});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_3D_gather_elements_3x2x2_indices_int32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2});
    int64_t axis = -1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{1, 2,
                              3, 4,

                              5, 6,
                              7, 8,

                              9, 10,
                              11, 12};
    std::vector<int32_t> indices{1, 0,
                                 0, 1,

                                 1, 1,
                                 1, 0,

                                 0, 0,
                                 1, 1};

    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{2, 1,
                                                           3, 4,

                                                           6, 6,
                                                           8, 7,

                                                           9, 9,
                                                           12, 12});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_4D_gather_elements_3x2x2x2_indices_int64)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i64, PartialShape{3, 2, 2, 4});
    int64_t axis = -1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{ 1,  2,
                               3,  4,

                               5,  6,
                               7,  8,


                               9, 10,
                              11, 12,

                              13, 14,
                              15, 16,


                              17, 18,
                              19, 20,

                              21, 22,
                              23, 24};
    std::vector<int64_t> indices{1, 0, 0, 0,
                                 0, 1, 1, 0,

                                 1, 1, 1, 1,
                                 1, 0, 0, 1,


                                 0, 0, 0, 1,
                                 1, 1, 1, 0,

                                 0, 0, 0, 0,
                                 1, 0, 1, 0,


                                 1, 1, 1, 1,
                                 1, 0, 1, 0,

                                 1, 0, 0, 1,
                                 0, 0, 0, 0};

    test_case.add_input<int32_t>(data);
    test_case.add_input<int64_t>(indices);
    test_case.add_expected_output<int32_t>(vector<int32_t>{2, 1, 1, 1,
                                                           3, 4, 4, 3,

                                                           6, 6, 6, 6,
                                                           8, 7, 7, 8,


                                                           9, 9, 9, 10,
                                                           12, 12, 12, 11,

                                                           13, 13, 13, 13,
                                                           16, 15, 16, 15,


                                                           18, 18, 18, 18,
                                                           20, 19, 20, 19,

                                                           22, 21, 21, 22,
                                                           23, 23, 23, 23});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_3D_gather_elements_3x2x2_indices_int64)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i64, PartialShape{3, 2, 2});
    int64_t axis = -1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{1, 2,
                              3, 4,
                              5, 6,

                              7, 8,
                              9, 10,
                              11, 12};
    std::vector<int64_t> indices{1, 0,
                                 0, 1,
                                 1, 1,

                                 1, 0,
                                 0, 0,
                                 1, 1};

    test_case.add_input<int32_t>(data);
    test_case.add_input<int64_t>(indices);
    test_case.add_expected_output<int32_t>(vector<int32_t>{2, 1,
                                                           3, 4,
                                                           6, 6,

                                                           8, 7,
                                                           9, 9,
                                                           12, 12});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_3x2_data_bool)
{
    auto arg1 = make_shared<op::Parameter>(element::boolean, PartialShape{3, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<bool> data{true, false, true,
                           true, false, false};
    std::vector<int32_t> indices{0, 1,
                                 0, 2};

    test_case.add_input<bool>(data);
    test_case.add_input<int32_t>(indices);
    test_case.add_expected_output<bool>(vector<bool>{true, true,
                                                     true, false});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x3_data_float32)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape{3, 3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<float> data{1.0f, 2.0f, 3.0f,
                            4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f};
    std::vector<int32_t> indices{1, 2, 0,
                                 2, 0, 0};

    test_case.add_input<float>(data);
    test_case.add_input<int32_t>(indices);
    test_case.add_expected_output<float>(vector<float>{4.0f, 8.0f, 3.0f,
                                                       7.0f, 2.0f, 3.0f});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x2x1_data_float32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2, 1});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{4, 2, 1});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{5,
                              4,

                              1,
                              4};
    std::vector<int32_t> indices{0,
                                 0,
                                 1,
                                 1,

                                 1,
                                 1,
                                 0,
                                 1};

    test_case.add_input<int32_t>(data);
    test_case.add_input<int32_t>(indices);
    test_case.add_expected_output<int32_t>({5,
                                            4,
                                            1,
                                            4,

                                            1,
                                            4,
                                            5,
                                            4});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_1D_gather_elements_negative_test)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{7});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    std::vector<int32_t> data{1, 2, 3};
    std::vector<int32_t> indices{1, 2, 0, 2, 0, 0, 8};

    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{2, 3, 1, 3, 1, 1, 3});
    try
    {
        test_case.run();
        // Should have thrown, so fail if it didn't
        FAIL() << "Evaluate out ouf bound indices check failed";
    }
    catch (const std::domain_error& error)
    {
        ASSERT_EQ(error.what(), std::string("indices values of GatherElement exceed data size"));
    }
    catch (...)
    {
        FAIL() << "Evaluate out ouf bound indices check failed";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_negative_test)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    std::vector<int32_t> data{1, 2, 3,
                              4, 5, 6,
                              7, 8, 9};
    std::vector<int32_t> indices{1, 3, 0,
                                 2, 0, 0};

    test_case.add_multiple_inputs<int32_t>({data, indices});
    test_case.add_expected_output<int32_t>(vector<int32_t>{4, 8, 3,
                                                           7, 2, 3});
    // clang-format on
    try
    {
        test_case.run();
        // Should have thrown, so fail if it didn't
        FAIL() << "Evaluate out ouf bound indices check failed";
    }
    catch (const std::domain_error& error)
    {
        ASSERT_EQ(error.what(), std::string("indices values of GatherElement exceed data size"));
    }
    catch (...)
    {
        FAIL() << "Evaluate out ouf bound indices check failed";
    }
}
