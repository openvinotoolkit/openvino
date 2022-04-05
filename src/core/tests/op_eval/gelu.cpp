// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gelu.hpp"

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

TEST(op_eval, gelu_tanh) {
    auto p = make_shared<op::Parameter>(element::f32, Shape{});
    auto gelu = make_shared<op::v7::Gelu>(p, op::GeluApproximationMode::TANH);
    auto fun = make_shared<Function>(OutputVector{gelu}, ParameterVector{p});

    std::vector<std::vector<float>> inputs{{-1.0}, {-0.5}, {0}, {0.5}, {1.0}};
    std::vector<std::vector<float>> expected_result{{-0.15880796}, {-0.154286}, {0}, {0.345714}, {0.841192}};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        OPENVINO_SUPPRESS_DEPRECATED_START
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{}, inputs[i])}));
        OPENVINO_SUPPRESS_DEPRECATED_END
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), (Shape{}));
        auto result_data = read_vector<float>(result);
        EXPECT_NEAR(result_data[0], expected_result[i][0], 0.000001);
    }
}

TEST(op_eval, gelu_erf) {
    auto p = make_shared<op::Parameter>(element::f32, Shape{});
    auto gelu = make_shared<op::v7::Gelu>(p, op::GeluApproximationMode::ERF);
    auto fun = make_shared<Function>(OutputVector{gelu}, ParameterVector{p});

    std::vector<std::vector<float>> inputs{{-1.0}, {-0.5}, {0}, {0.5}, {1.0}};
    std::vector<std::vector<float>> expected_result{{-0.15865529}, {-0.15426877}, {0}, {0.34573123}, {0.8413447}};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        OPENVINO_SUPPRESS_DEPRECATED_START
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{}, inputs[i])}));
        OPENVINO_SUPPRESS_DEPRECATED_END
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), (Shape{}));
        auto result_data = read_vector<float>(result);
        EXPECT_NEAR(result_data[0], expected_result[i][0], 0.000001);
    }
}
