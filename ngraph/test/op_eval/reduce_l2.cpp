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

#include "ngraph/opsets/opset4.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, reduce_l2_one_axis_keep_dims)
{
    auto data = make_shared<opset4::Parameter>(element::f32, Shape{3, 2, 2});
    auto axes = opset4::Constant::create(element::i32, Shape{1}, {2});
    auto reduce = make_shared<op::v4::ReduceL2>(data, axes, true);
    auto fun = make_shared<Function>(OutputVector{reduce}, ParameterVector{data});

    std::vector<float> inputs{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> expected_result{
        2.23606798, 5.0, 7.81024968, 10.63014581, 13.45362405, 16.2788206};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{3, 2, 2}, inputs),
                               make_host_tensor<element::Type_t::i32>(Shape{1}, {2})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{std::vector<size_t>({3, 2, 1})});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

TEST(op_eval, reduce_l2_one_axis_do_not_keep_dims)
{
    auto data = make_shared<opset4::Parameter>(element::f32, Shape{3, 2, 2});
    auto axes = opset4::Constant::create(element::i32, Shape{1}, {2});
    auto reduce = make_shared<op::v4::ReduceL2>(data, axes, false);
    auto fun = make_shared<Function>(OutputVector{reduce}, ParameterVector{data});

    std::vector<float> inputs{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> expected_result{
        2.23606798, 5.0, 7.81024968, 10.63014581, 13.45362405, 16.2788206};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{3, 2, 2}, inputs),
                               make_host_tensor<element::Type_t::i32>(Shape{1}, {2})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{std::vector<size_t>({3, 2})});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
