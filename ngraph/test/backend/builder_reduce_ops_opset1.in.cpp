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

#include <numeric>

#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;
using namespace ngraph::test;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean)
{
    const Shape input_shape{4, 3, 2};
    const AxisSet axes{1, 2};
    const auto input = make_shared<op::Parameter>(element::Type_t::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);
    auto function = make_shared<Function>(mean_builder, ParameterVector{input});

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    vector<float> input_values(shape_size(input_shape));
    iota(begin(input_values), end(input_values), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{4}, vector<float>{2.5f, 8.5f, 14.5f, 20.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean_dynamic)
{
    const Shape input_shape{2, 4, 5};
    const AxisSet axes{0, 1};
    const auto input = make_shared<op::Parameter>(element::Type_t::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);
    auto function = make_shared<Function>(mean_builder, ParameterVector{input});

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    vector<float> input_values(shape_size(input_shape));
    iota(begin(input_values), end(input_values), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{5},
                                         vector<float>{17.5f, 18.5f, 19.5f, 20.5f, 21.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean_dynamic_2)
{
    const Shape input_shape{2, 1, 3};
    const AxisSet axes{1, 2};
    const auto input = make_shared<op::Parameter>(element::Type_t::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);
    auto function = make_shared<Function>(mean_builder, ParameterVector{input});

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(function);
    vector<float> input_values(shape_size(input_shape));
    iota(begin(input_values), end(input_values), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{2}, vector<float>{1.f, 4.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_collapse_5d_to_3d)
{
    Shape shape_input{1, 2, 3, 4, 5};
    Shape shape_r{1, 24, 5};

    const auto elems_in_tensor = shape_size(shape_input);

    const auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_input);
    const auto builder_collapse = builder::opset1::collapse(A, 1, shape_input.size() - 2);
    const auto f = make_shared<Function>(builder_collapse, ParameterVector{A});

    vector<float> a(elems_in_tensor, 1);
    vector<float> b(elems_in_tensor, 1);

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>(shape_input, {a});
    test_case.add_expected_output<float>(shape_r, b);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_collapse_all_dims)
{
    Shape shape_input{1, 2, 3, 4, 5, 6};
    Shape shape_r{720};

    const auto elems_in_tensor = shape_size(shape_input);

    const auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_input);
    const auto builder_collapse = builder::opset1::collapse(A, 0, shape_input.size() - 1);
    const auto f = make_shared<Function>(builder_collapse, ParameterVector{A});

    vector<float> a(elems_in_tensor, 1);
    vector<float> b(elems_in_tensor, 1);

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>(shape_input, {a});
    test_case.add_expected_output<float>(shape_r, b);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_collapse_none)
{
    Shape shape_input{1, 2, 3, 4, 5, 6};

    const auto elems_in_tensor = shape_size(shape_input);

    const auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_input);
    const auto builder_collapse = builder::opset1::collapse(A, 2, shape_input.size() - 4);
    const auto f = make_shared<Function>(builder_collapse, ParameterVector{A});

    vector<float> a(elems_in_tensor, 1);
    vector<float> b(elems_in_tensor, 1);

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>(shape_input, {a});
    test_case.add_expected_output<float>(shape_input, b);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_collapse_dyn_shape)
{
    PartialShape pshape_input{1, 2, 3, 4, 5, Dimension()};
    PartialShape pshape_output{1, 24, 5, Dimension()};

    const auto A = make_shared<op::Parameter>(element::Type_t::f32, pshape_input);
    EXPECT_TRUE(A->get_output_partial_shape(0).same_scheme(
        PartialShape{1, 2, 3, 4, 5, Dimension::dynamic()}));
    const auto builder_collapse = builder::opset1::collapse(A, 1, 3);
    const auto f = make_shared<Function>(builder_collapse, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine, TestCaseType::DYNAMIC>(f);

    const size_t NUM_DIMENSIONS_TO_TEST = 5;
    for (size_t dim = 1; dim < NUM_DIMENSIONS_TO_TEST; dim++)
    {
        Shape shape_input{1, 2, 3, 4, 5, dim};
        Shape shape_output{1, 24, 5, dim};
        const auto elems_in_tensor = shape_size(shape_input);

        std::vector<float> input_values(elems_in_tensor, 1);
        std::vector<float> expected_values(elems_in_tensor, 1);

        test_case.add_input<float>(shape_input, {input_values});
        test_case.add_expected_output<float>(shape_output, expected_values);
        test_case.run();
    }
}
