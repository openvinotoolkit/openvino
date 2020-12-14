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

    std::vector<int32_t> data{1, 2, 3};
    std::vector<int32_t> indices{1, 2, 0, 2, 0, 0, 2};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_multiple_inputs<int32_t>({data, indices});
    test_case1.add_expected_output<int32_t>(vector<int32_t>{2, 3, 1, 3, 1, 1, 3});
    test_case1.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x2_indices_int32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<int32_t> data{1, 2, 3, 4};
    std::vector<int32_t> indices{0, 1, 0, 0};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_multiple_inputs<int32_t>({data, indices});
    test_case1.add_expected_output<int32_t>(vector<int32_t>{1, 4, 1, 2});
    test_case1.run();

    axis = 1;
    gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case2 = test::TestCase<TestEngine>(fun);
    test_case2.add_multiple_inputs<int32_t>({data, indices});
    test_case2.add_expected_output<int32_t>(vector<int32_t>{1, 2, 3, 3});
    test_case2.run();

    axis = -1;
    gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});
    auto test_case3 = test::TestCase<TestEngine>(fun);
    test_case3.add_multiple_inputs<int32_t>({data, indices});
    test_case3.add_expected_output<int32_t>(vector<int32_t>{1, 2, 3, 3});
    test_case3.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x3_indices_int32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int32_t> indices{1, 2, 0, 2, 0, 0};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_multiple_inputs<int32_t>({data, indices});
    test_case1.add_expected_output<int32_t>(vector<int32_t>{4, 8, 3, 7, 2, 3});
    test_case1.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_3D_gather_elements_3x2x2_indices_int32)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2});
    int64_t axis = -1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> indices{1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_multiple_inputs<int32_t>({data, indices});
    test_case1.add_expected_output<int32_t>(vector<int32_t>{2, 1, 3, 4, 6, 6, 8, 7, 9, 9, 12, 12});
    test_case1.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_3D_gather_elements_3x2x2_indices_int64)
{
    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape{3, 2, 2});
    auto arg2 = make_shared<op::Parameter>(element::i64, PartialShape{3, 2, 2});
    int64_t axis = -1;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int64_t> indices{1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_input<int32_t>(data);
    test_case1.add_input<int64_t>(indices);
    test_case1.add_expected_output<int32_t>(vector<int32_t>{2, 1, 3, 4, 6, 6, 8, 7, 9, 9, 12, 12});
    test_case1.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_3x2_data_bool)
{
    auto arg1 = make_shared<op::Parameter>(element::boolean, PartialShape{3, 2});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 2});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<bool> data{true, false, true, true, false, false};
    std::vector<int32_t> indices{0, 1, 0, 2};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_input<bool>(data);
    test_case1.add_input<int32_t>(indices);
    test_case1.add_expected_output<bool>(vector<bool>{true, true, true, false});
    test_case1.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_2D_gather_elements_2x3_data_float32)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape{3, 3});
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape{2, 3});
    int64_t axis = 0;

    auto gather_el = make_shared<op::v6::GatherElements>(arg1, arg2, axis);
    auto fun = make_shared<Function>(OutputVector{gather_el}, ParameterVector{arg1, arg2});

    std::vector<float_t> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<int32_t> indices{1, 2, 0, 2, 0, 0};

    auto test_case1 = test::TestCase<TestEngine>(fun);
    test_case1.add_input<float_t>(data);
    test_case1.add_input<int32_t>(indices);
    test_case1.add_expected_output<float_t>(vector<float_t>{4.0f, 8.0f, 3.0f, 7.0f, 2.0f, 3.0f});
    test_case1.run();
}
