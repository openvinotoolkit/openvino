// Copyright (C) 2018-2022 Intel Corporation
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

TEST(op_eval, softsign) {
    auto p = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{4});
    auto softsign = std::make_shared<ov::intel_gna::op::SoftSign>(p);
    auto fun = std::make_shared<ngraph::Function>(ngraph::OutputVector{softsign}, ngraph::ParameterVector{p});

    float inputs[] = {-1.0, 0.0, 1.0, 20.0};
    std::vector<float> expected_result{-0.5, 0.0, 0.5, 0.952381};

    ov::TensorVector result(1);
    ov::Tensor input{ov::element::f32, ov::Shape{4}, inputs};

    ASSERT_TRUE(fun->evaluate(result, ov::TensorVector{input}));

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].get_element_type(), ngraph::element::f32);
    EXPECT_EQ(result[0].get_shape(), ngraph::Shape{4});
    EXPECT_EQ(result[0].get_size(), 4);

    const float * result_data = result[0].data<float>();
    for (size_t i = 0; i < result[0].get_size(); ++i)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

