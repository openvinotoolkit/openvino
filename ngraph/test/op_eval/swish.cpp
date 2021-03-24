// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/swish.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, swish_with_beta1)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{});
    auto swish = make_shared<op::v4::Swish>(p, beta);
    auto fun = make_shared<Function>(OutputVector{swish}, ParameterVector{p, beta});

    std::vector<float> inputs{-0.5, 0.0, 0.5};
    std::vector<float> expected_result{-0.18877034, 0.0, 0.31122968};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs),
                               make_host_tensor<element::Type_t::f32>(Shape{}, {1.0})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{3});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

TEST(op_eval, swish_with_beta0_75)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{});
    auto swish = make_shared<op::v4::Swish>(p, beta);
    auto fun = make_shared<Function>(OutputVector{swish}, ParameterVector{p, beta});

    std::vector<float> inputs{-0.5, 0.0, 0.5};
    std::vector<float> expected_result{-0.2036667, 0.0, 0.2963333};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs),
                               make_host_tensor<element::Type_t::f32>(Shape{}, {0.75})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{3});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

TEST(op_eval, swish_without_beta)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{3});
    auto swish = make_shared<op::v4::Swish>(p);
    auto fun = make_shared<Function>(OutputVector{swish}, ParameterVector{p});

    std::vector<float> inputs{-0.5, 0.0, 0.5};
    std::vector<float> expected_result{-0.18877034, 0.0, 0.31122968};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{3});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
