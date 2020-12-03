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

NGRAPH_TEST(${BACKEND_NAME}, transpose)
{
    //
    // Create a graph for f(x,perm) = Transpose(x,Convert<i64>(perm)). We'll do the permutation in
    // i32 and cast it to i64, just for fun (and to mirror the TensorFlow test I am porting here).
    //
    auto x = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto perm =
        make_shared<op::Parameter>(element::Type_t::i32, PartialShape{Dimension::dynamic()});
    auto perm_i64 = make_shared<op::Convert>(perm, element::Type_t::i64);

    auto x_transpose = make_shared<op::Transpose>(x, perm_i64);

    auto f = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{x, perm});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::Type_t::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{2, 3}, Shape{2, 2, 3}};
    std::vector<std::vector<int32_t>> perms{{0, 1}, {1, 0}, {2, 1, 0}};
    std::vector<std::vector<float>> inputs{
        {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> expected_result_shapes{Shape{2, 3}, Shape{3, 2}, {3, 2, 2}};
    // Generated with numpy, so don't worry. :)
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::Type_t::f32, x_shapes[i]);
        auto t_perm = backend->create_tensor(element::Type_t::i32, Shape{perms[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_perm, perms[i]);

        ex->call_with_validate({t_r}, {t_x, t_perm});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}
