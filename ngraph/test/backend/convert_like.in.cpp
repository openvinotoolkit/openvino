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

NGRAPH_TEST(${BACKEND_NAME}, convert_like_float32_int32)
{
    Shape input_shape{2, 3, 1};
    const auto data = make_shared<op::Parameter>(element::f32, input_shape);
    const auto like = make_shared<op::Parameter>(element::i32, input_shape);
    const auto convert_like = make_shared<op::v1::ConvertLike>(data, like);
    const auto f = make_shared<Function>(convert_like, ParameterVector{data, like});

    vector<float> data_vect = {-1.8, 0.2f, 1.4f, 2.1f, 3.9f, 4.3f};
    vector<int32_t> like_vect(shape_size(input_shape), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(input_shape, data_vect);
    test_case.add_input<int32_t>(input_shape, like_vect);
    test_case.add_expected_output<int>(input_shape, {-1, 0, 1, 2, 3, 4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_int32_float32)
{
    Shape shape{2, 2};
    const auto data = make_shared<op::Parameter>(element::i32, shape);
    const auto like = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like),
                                   ParameterVector{data, like});

    vector<int32_t> data_vect{281, 2, 3, 4};
    vector<float> like_vect(shape_size(shape), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>(shape, data_vect);
    test_case.add_input<float>(shape, like_vect);
    test_case.add_expected_output<float>(shape, {281, 2, 3, 4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_uint16_float32)
{
    Shape shape{2, 2};
    const auto data = make_shared<op::Parameter>(element::u16, shape);
    const auto like = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like),
                                   ParameterVector{data, like});

    vector<uint16_t> data_vect{1, 2, 3, 4};
    vector<float> like_vect(shape_size(shape), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint16_t>(shape, data_vect);
    test_case.add_input<float>(shape, like_vect);
    test_case.add_expected_output<float>(shape, {1, 2, 3, 4});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_int32_bool)
{
    Shape shape{2, 3};
    const auto data = make_shared<op::Parameter>(element::i32, shape);
    const auto like = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like),
                                   ParameterVector{data, like});

    int32_t lowest = std::numeric_limits<int32_t>::lowest();
    int32_t max = std::numeric_limits<int32_t>::max();

    vector<int32_t> data_vect{0, 12, 23, 0, lowest, max};
    vector<char> like_vect(shape_size(shape), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>(shape, data_vect);
    test_case.add_input<char>(shape, like_vect);
    test_case.add_expected_output<char>(shape, {0, 1, 1, 0, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_float32_bool)
{
    Shape shape{3, 3};
    const auto data = make_shared<op::Parameter>(element::f32, shape);
    const auto like = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like),
                                   ParameterVector{data, like});

    float lowest = std::numeric_limits<float>::lowest();
    float max = std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::min();
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();

    vector<float> data_vect{0.f, 1.5745f, 0.12352f, 0.f, lowest, max, min, pos_inf, neg_inf};
    vector<char> like_vect(shape_size(shape), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, data_vect);
    test_case.add_input<char>(shape, like_vect);
    test_case.add_expected_output<char>(shape, {0, 1, 1, 0, 1, 1, 1, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_float32_bfloat16)
{
    Shape shape{1, 1, 3, 5};
    const auto data = make_shared<op::Parameter>(element::f32, shape);
    const auto like = make_shared<op::Parameter>(element::bf16, shape);
    auto f = make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like),
                                   ParameterVector{data, like});

    vector<float> data_vect{
        0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f};
    vector<bfloat16> like_vect(shape_size(shape), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, data_vect);
    test_case.add_input<bfloat16>(shape, like_vect);
    test_case.add_expected_output<bfloat16>(
        shape,
        vector<bfloat16>{
            0.5, 1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 1.5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_bfloat16_float32)
{
    Shape shape_data{1, 1, 3, 5};
    Shape shape_like{4};
    const auto data = make_shared<op::Parameter>(element::bf16, shape_data);
    const auto like = make_shared<op::Parameter>(element::f32, shape_like);
    auto f = make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like),
                                   ParameterVector{data, like});

    vector<bfloat16> data_vect{
        0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f};
    vector<float> like_vect(shape_size(shape_like), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<bfloat16>(shape_data, data_vect);
    test_case.add_input<float>(shape_like, like_vect);
    test_case.add_expected_output<float>(
        shape_data, {0.5, 1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 1.5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, convert_like_dyn_float16_to_int64)
{
    PartialShape pshape_data{Dimension::dynamic(), 2, 2, Dimension::dynamic()};
    Shape shape_like{};
    const auto data = make_shared<op::Parameter>(element::f16, pshape_data);
    const auto like = op::Constant::create(element::i64, Shape{}, {0});
    auto f =
        make_shared<Function>(make_shared<op::v1::ConvertLike>(data, like), ParameterVector{data});

    vector<float16> data_vect = {-3.21f, 0.1f, 2.6f, 4.99f};
    Shape shape_data{1, 2, 2, 1};

    auto test_case = test::TestCase<TestEngine, ngraph::test::TestCaseType::DYNAMIC>(f);
    test_case.add_input<float16>(shape_data, data_vect);
    test_case.add_expected_output<int64_t>(shape_data, vector<int64_t>{-3, 0, 2, 4});
    test_case.run();
}
