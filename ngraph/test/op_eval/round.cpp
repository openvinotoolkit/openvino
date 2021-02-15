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

#include "ngraph/op/round.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, rounding_to_even)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{9});
    auto round = make_shared<op::v5::Round>(p, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});

    std::vector<float> inputs{-2.5f, -1.5f, -0.5f, 0.5f, 0.9f, 1.5f, 2.3f, 2.5f, 3.5f};
    std::vector<float> expected_result{-2.f, -2.f, -0.f, 0.f, 1.f, 2.f, 2.f, 2.f, 4.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{9}, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{9});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

TEST(op_eval, rounding_away)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{9});
    auto round = make_shared<op::v5::Round>(p, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});

    std::vector<float> inputs{-2.5f, -1.5f, -0.5f, 0.5f, 0.9f, 1.5f, 2.3f, 2.5f, 3.5f};
    std::vector<float> expected_result{-3.f, -2.f, -1.f, 1.f, 1.f, 2.f, 2.f, 3.f, 4.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{9}, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{9});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
