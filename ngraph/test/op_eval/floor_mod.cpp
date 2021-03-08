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

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/floor_mod.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, floor_mod)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{4});
    auto b = make_shared<op::Parameter>(element::f32, Shape{4});
    auto floor_mod = make_shared<op::v1::FloorMod>(a, b);
    auto fun = make_shared<Function>(OutputVector{floor_mod}, ParameterVector{a, b});

    std::vector<float> a_value{5.1, -5.1, 5.1, -5.1};
    std::vector<float> b_value{3.0, 3.0, -3.0, -3.0};
    std::vector<float> expected_result{2.1, 0.9, -0.9, -2.1};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{4}, a_value),
                               make_host_tensor<element::Type_t::f32>(Shape{4}, b_value)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{4});
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
