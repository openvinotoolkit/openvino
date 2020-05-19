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
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, dyn_broadcast)
{
    // Create a graph for
    //   f(x,shape:i32,axes:32) = Broadcast(x,Convert<i64>(shape),Convert<i64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto shape = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto shape_i64 = make_shared<op::Convert>(shape, element::i64);
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto bc = make_shared<op::DynBroadcast>(x, shape_i64, axes_i64);

    auto f = make_shared<Function>(NodeVector{bc}, ParameterVector{x, shape, axes});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{}, Shape{}, Shape{2}, Shape{2}};
    std::vector<std::vector<int32_t>> shapes{{2, 2}, {2, 2, 2}, {3, 2}, {2, 3}};
    std::vector<std::vector<int32_t>> axeses{{0, 1}, {0, 1, 2}, {0}, {1}};
    std::vector<std::vector<float>> inputs{{6}, {7}, {10, 11}, {10, 11}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 2}, Shape{2, 2, 2}, Shape{3, 2}, Shape{2, 3}};
    std::vector<std::vector<float>> expected_results{
        {6, 6, 6, 6}, {7, 7, 7, 7, 7, 7, 7, 7}, {10, 11, 10, 11, 10, 11}, {10, 10, 10, 11, 11, 11}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_shape = backend->create_tensor(element::i32, Shape{shapes[i].size()});
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_shape, shapes[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_shape, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_v1)
{
    // Create a graph for
    //   f(x,shape:i32,axes:32) = Broadcast(x,Convert<i64>(shape),Convert<i64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto shape = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto shape_i64 = make_shared<op::Convert>(shape, element::i64);
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto bc = make_shared<op::v1::Broadcast>(x, shape_i64, axes_i64);

    auto f = make_shared<Function>(NodeVector{bc}, ParameterVector{x, shape, axes});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{}, Shape{}, Shape{2}, Shape{2}};
    std::vector<std::vector<int32_t>> shapes{{2, 2}, {2, 2, 2}, {3, 2}, {2, 3}, {2, 2}, {2, 2}};
    std::vector<std::vector<int32_t>> axeses{{}, {}, {1}, {0}, {0}, {1}};
    std::vector<std::vector<float>> inputs{{6}, {7}, {10, 11}, {10, 11}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 2}, Shape{2, 2, 2}, Shape{3, 2}, Shape{2, 3}, Shape{2, 2}, Shape{2, 2}};
    std::vector<std::vector<float>> expected_results{{6, 6, 6, 6},
                                                     {7, 7, 7, 7, 7, 7, 7, 7},
                                                     {10, 11, 10, 11, 10, 11},
                                                     {10, 10, 10, 11, 11, 11},
                                                     {10, 10, 11, 11},
                                                     {10, 11, 10, 11}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_shape = backend->create_tensor(element::i32, Shape{shapes[i].size()});
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_shape, shapes[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_shape, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}
