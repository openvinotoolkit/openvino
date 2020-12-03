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
#include "util/type_prop.hpp"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

//
// Graph before (params in [] brackets, constants in () parens, results in {} braces):
//
//  [x]   [y]   [z]
//    \   /      |
//     Add   (k) |
//       \   /   |
//        Mul**  |
//          \   /
//           Sub
//            |
//           {r}
//
// Param substitutions:
//
//    [x] -> [x']
//
// Body substitutions:
//
//    (k) -> (k')
//    [y] -> (k'')
//    [z] -> [x'] + **
//
// After replacement:
//
//  [x']---------
//    |          |
//    |   (k'')  |    [z] and [y] is still there, but dead
//    \   /      |
//     Add  (k') |
//       \   /   |
//        Mul    |
//          \   /
//           Sub ***
//            |
//           {r}
//
TEST(replace_node, replace_nodes)
{
    auto x = make_shared<op::Parameter>(element::f32, Shape{2});
    auto y = make_shared<op::Parameter>(element::f32, Shape{2});
    auto z = make_shared<op::Parameter>(element::f32, Shape{2});

    auto add = make_shared<op::v1::Add>(x, y);
    auto k = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{1, 2});
    auto mul = make_shared<op::v1::Multiply>(add, k);
    auto sub = make_shared<op::v1::Subtract>(mul, z);

    auto f = make_shared<Function>(NodeVector{sub}, ParameterVector{x, y, z});

    unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Parameter>> parameter_replacement_map;
    auto x_replacement = make_shared<op::Parameter>(element::f32, Shape{2});
    parameter_replacement_map[x] = x_replacement;

    unordered_map<shared_ptr<Node>, shared_ptr<Node>> body_replacement_map;
    auto y_replacement = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{3, 4});
    auto k_replacement = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{5, 6});
    auto z_replacement = make_shared<op::v1::Add>(x_replacement, mul);
    body_replacement_map[y] = y_replacement;
    body_replacement_map[k] = k_replacement;
    body_replacement_map[z] = z_replacement;

    replace_nodes(f, parameter_replacement_map, body_replacement_map);

    // Should still have three params.
    ASSERT_EQ(f->get_parameters().size(), 3);

    // The three params be {x_replacement, y, z}.
    ASSERT_EQ(f->get_parameters()[0], x_replacement);
    ASSERT_EQ(f->get_parameters()[1], y);
    ASSERT_EQ(f->get_parameters()[2], z);

    // y, z should be dead.
    ASSERT_EQ(y->get_users(true).size(), 0);
    ASSERT_EQ(z->get_users(true).size(), 0);

    // Should still have one result.
    ASSERT_EQ(f->get_results().size(), 1);

    // Result node should be sub (unchanged).
    ASSERT_EQ(f->get_results()[0]->get_input_node_shared_ptr(0), sub);

    // sub's arguments should be mul (unchanged) and z_replacement.
    ASSERT_EQ(sub->get_input_node_shared_ptr(0), mul);
    ASSERT_EQ(sub->get_input_node_shared_ptr(1), z_replacement);

    // mul's arguments should be add (unchanged) and k_replacement.
    ASSERT_EQ(mul->get_input_node_shared_ptr(0), add);
    ASSERT_EQ(mul->get_input_node_shared_ptr(1), k_replacement);

    // add's arguments should be x_replacement and y_replacement.
    ASSERT_EQ(add->get_input_node_shared_ptr(0), x_replacement);
    ASSERT_EQ(add->get_input_node_shared_ptr(1), y_replacement);

    // z_replacement's arguments should be x_replacement and mul.
    ASSERT_EQ(z_replacement->get_input_node_shared_ptr(0), x_replacement);
    ASSERT_EQ(z_replacement->get_input_node_shared_ptr(1), mul);
}
