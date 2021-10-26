// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ops/softsign.hpp"

#include <string>
#include <vector>

#include "execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/opsets/opset8.hpp"

using namespace GNAPluginNS;

TEST(op_eval, softsign) {
    auto p = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{4});
    auto softsign = std::make_shared<SoftSign>(p);
    auto fun = std::make_shared<ngraph::Function>(ngraph::OutputVector{softsign}, ngraph::ParameterVector{p});

    std::vector<float> inputs{-1.0, 0.0, 1.0, 20.0};
    std::vector<float> expected_result{0.5, 1.0, 0.5, 0.047619};

    auto result = std::make_shared<ngraph::HostTensor>();

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<ngraph::element::Type_t::f32>(ngraph::Shape{4}, inputs)}));
    OPENVINO_SUPPRESS_DEPRECATED_END

    EXPECT_EQ(result->get_element_type(), ngraph::element::f32);
    EXPECT_EQ(result->get_shape(), ngraph::Shape{4});
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
