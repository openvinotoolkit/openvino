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

#include <memory>

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(input_output, param_tensor)
{
    // Params have no arguments, so we can check that the value becomes a tensor output
    auto& et = element::f32;
    Shape shape{2, 4};
    auto param = make_shared<op::Parameter>(et, shape);

    ASSERT_EQ(param->get_output_size(), 1);
    ASSERT_EQ(et, param->get_element_type());
    ASSERT_EQ(shape, param->get_shape());
}

TEST(input_output, simple_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto add = make_shared<op::v1::Add>(param_0, param_1);

    // Sort the ops
    vector<shared_ptr<Node>> nodes;
    nodes.push_back(param_0);
    nodes.push_back(param_1);
    nodes.push_back(add);

    // At this point, the add should have each input associated with the output of the appropriate
    // parameter
    ASSERT_EQ(1, add->get_output_size());
    ASSERT_EQ(2, add->get_input_size());
    for (size_t i = 0; i < add->get_input_size(); i++)
    {
        ASSERT_EQ(add->input_value(i).get_node_shared_ptr(), nodes.at(i));
    }
}
