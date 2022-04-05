// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/round.hpp"

#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, rounding_to_even) {
    auto p = make_shared<op::Parameter>(element::f32, Shape{9});
    auto round = make_shared<op::v5::Round>(p, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});

    std::vector<float> inputs{-2.5f, -1.5f, -0.5f, 0.5f, 0.9f, 1.5f, 2.3f, 2.5f, 3.5f};
    std::vector<float> expected_result{-2.f, -2.f, -0.f, 0.f, 1.f, 2.f, 2.f, 2.f, 4.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{9}, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{9});
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

TEST(op_eval, rounding_away) {
    auto p = make_shared<op::Parameter>(element::f32, Shape{9});
    auto round = make_shared<op::v5::Round>(p, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});

    std::vector<float> inputs{-2.5f, -1.5f, -0.5f, 0.5f, 0.9f, 1.5f, 2.3f, 2.5f, 3.5f};
    std::vector<float> expected_result{-3.f, -2.f, -1.f, 1.f, 1.f, 2.f, 2.f, 3.f, 4.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{9}, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{9});
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
