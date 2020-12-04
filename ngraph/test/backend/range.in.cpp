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
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::test;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename T>
struct RangeTest
{
    T start;
    T stop;
    T step;
    Shape expected_result_shape;
    std::vector<T> expected_result;
};

// ------------------------------ V0 ------------------------------

// TODO(amprocte): We should test this with more than just int32, but there is a bug in the
// handling of element type-changing that is currently blocking doing that easily.
NGRAPH_TEST(${BACKEND_NAME}, range)
{
    // Create a graph for f(start,stop,step) = Range(start,stop,step).
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);
    ASSERT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));

    auto f = make_shared<Function>(NodeVector{range}, ParameterVector{start, stop, step});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    std::vector<RangeTest<int32_t>> int32_tests = {
        RangeTest<int32_t>{0, 10, 1, Shape{10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
        RangeTest<int32_t>{-5, 6, 3, Shape{4}, {-5, -2, 1, 4}},
        RangeTest<int32_t>{10, 0, 1, Shape{0}, {}},
        RangeTest<int32_t>{10, 5, -3, Shape{2}, {10, 7}}};

    for (auto& test : int32_tests)
    {
        auto t_start = backend->create_tensor(element::i32, Shape{});
        auto t_stop = backend->create_tensor(element::i32, Shape{});
        auto t_step = backend->create_tensor(element::i32, Shape{});

        copy_data(t_start, std::vector<int32_t>{test.start});
        copy_data(t_stop, std::vector<int32_t>{test.stop});
        copy_data(t_step, std::vector<int32_t>{test.step});

        ex->call_with_validate({t_r}, {t_start, t_stop, t_step});

        ASSERT_EQ(t_r->get_element_type(), element::i32);
        ASSERT_EQ(t_r->get_shape(), test.expected_result_shape);

        auto results = read_vector<int32_t>(t_r);

        ASSERT_EQ(results, test.expected_result);
    }
}

// ------------------------------ V4 ------------------------------

NGRAPH_TEST(${BACKEND_NAME}, range_v4_trunc_inputs)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Parameter>(element::f32, Shape{});

    auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto f = make_shared<Function>(range, ParameterVector{start, stop, step});

    std::vector<float> start_vect{1.2};
    std::vector<float> stop_vect{11.3};
    std::vector<float> step_vect{1.6f};

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(f);
    test_case.add_input<float>(Shape{}, start_vect);
    test_case.add_input<float>(Shape{}, stop_vect);
    test_case.add_input<float>(Shape{}, step_vect);
    test_case.add_expected_output<int32_t>(Shape{10},
                                           std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, range_v4_int32)
{
    element::Type_t et = element::i32;
    std::vector<RangeTest<int32_t>> int32_tests = {
        RangeTest<int32_t>{0, 10, 1, Shape{10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
        RangeTest<int32_t>{-5, 6, 3, Shape{4}, {-5, -2, 1, 4}},
        RangeTest<int32_t>{10, 0, 1, Shape{0}, {}},
        RangeTest<int32_t>{10, 5, -3, Shape{2}, {10, 7}}};

    for (auto& test : int32_tests)
    {
        auto start = make_shared<op::Constant>(et, Shape{}, std::vector<int32_t>{test.start});
        auto stop = make_shared<op::Constant>(et, Shape{}, std::vector<int32_t>{test.stop});
        auto step = make_shared<op::Constant>(et, Shape{}, std::vector<int32_t>{test.step});
        auto range = make_shared<op::v4::Range>(start, stop, step, et);
        auto pshape_out = range->get_output_partial_shape(0);
        ASSERT_TRUE(pshape_out.rank().is_static() && pshape_out.rank() == Dimension{1});
        auto f = make_shared<Function>(NodeVector{range}, ParameterVector{});

        auto test_case = test::TestCase<TestEngine>(f);

        test_case.add_expected_output<int32_t>(test.expected_result_shape, test.expected_result);
        test_case.run();
    }
}

NGRAPH_TEST(${BACKEND_NAME}, range_v4_float32)
{
    element::Type_t et = element::f32;
    std::vector<RangeTest<float>> float32_tests = {
        RangeTest<float>{0, 1, 0.25, Shape{4}, {0, 0.25, 0.5, 0.75}},
        RangeTest<float>{
            -1, 0.875, 0.2, Shape{10}, {-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8}},
        RangeTest<float>{10, 0, 1, Shape{0}, {}},
        RangeTest<float>{2, 0, -0.25, Shape{8}, {2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25}}};

    for (auto& test : float32_tests)
    {
        auto start = make_shared<op::Constant>(et, Shape{}, std::vector<float>{test.start});
        auto stop = make_shared<op::Constant>(et, Shape{}, std::vector<float>{test.stop});
        auto step = make_shared<op::Constant>(et, Shape{}, std::vector<float>{test.step});
        auto range = make_shared<op::v4::Range>(start, stop, step, et);
        auto pshape_out = range->get_output_partial_shape(0);
        ASSERT_TRUE(pshape_out.rank().is_static() && pshape_out.rank() == Dimension{1});
        auto f = make_shared<Function>(NodeVector{range}, ParameterVector{});

        auto test_case = test::TestCase<TestEngine>(f);

        test_case.add_expected_output<float>(test.expected_result_shape, test.expected_result);
        test_case.run();
    }
}
