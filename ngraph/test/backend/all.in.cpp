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
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, all_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 0, 1});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2_to_scalar_false)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 0, 1});
    auto result = backend->create_tensor(element::boolean, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2_to_scalar_true)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 1, 1, 1});
    auto result = backend->create_tensor(element::boolean, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x0_to_scalar)
{
    Shape shape{2, 0};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    auto result = backend->create_tensor(element::boolean, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x3_eliminate_col_dim)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, test::NDArray<char, 2>({{1, 0, 1}, {1, 1, 1}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x3_eliminate_row_dim)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, test::NDArray<char, 2>({{1, 0, 1}, {1, 1, 0}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{1, 0, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dim_0)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2, 3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 0, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dim_1)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2, 3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{1, 0, 0, 0, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dim_2)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2, 2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dims_0_1)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 0, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dims_0_2)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dims_1_2)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_2x2x3_eliminate_dims_0_1_2)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::All>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(
        a, test::NDArray<char, 3>({{{1, 0, 1}, {1, 1, 0}}, {{0, 1, 0}, {1, 1, 1}}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_dynamic_axis)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = op::Constant::create(element::i64, Shape{1}, {1});
    auto f = make_shared<Function>(make_shared<op::All>(A, B), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, test::NDArray<char, 2>({{1, 0, 1}, {1, 1, 1}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_change_axis)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = op::Constant::create(element::i64, Shape{1}, {1});
    auto all = make_shared<op::All>(A, B);
    ASSERT_EQ(all->get_reduction_axes(), AxisSet{1});
    auto f = make_shared<Function>(all, ParameterVector{A});

    all->set_reduction_axes(AxisSet{0});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, test::NDArray<char, 2>({{1, 0, 1}, {1, 1, 1}}).get_vector());
    auto result = backend->create_tensor(element::boolean, Shape{3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<char>{1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, all_dynamic)
{
    // Create a graph for f(x,axes:int32) = All(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto all = make_shared<op::All>(x, axes_i64);
    ASSERT_TRUE(all->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{all}, ParameterVector{x, axes});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::boolean, PartialShape::dynamic());

    std::vector<Shape> x_shapes{
        Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{5}, Shape{5}};
    std::vector<std::vector<int32_t>> axeses{{}, {0}, {1}, {0, 1}, {}, {0}};
    std::vector<std::vector<char>> inputs{{1, 0, 1, 0, 1, 0},
                                          {1, 0, 1, 0, 0, 1},
                                          {1, 0, 1, 1, 1, 1},
                                          {1, 0, 1, 0, 1, 0},
                                          {1, 0, 1, 0, 1},
                                          {1, 0, 1, 0, 1}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 3}, Shape{3}, Shape{2}, Shape{}, Shape{5}, Shape{}};
    std::vector<std::vector<char>> expected_results{
        {1, 0, 1, 0, 1, 0}, {0, 0, 1}, {0, 1}, {0}, {1, 0, 1, 0, 1}, {0}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::boolean, x_shapes[i]);
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<char>(t_r);

        ASSERT_EQ(results, expected_results[i]);
    }
}
