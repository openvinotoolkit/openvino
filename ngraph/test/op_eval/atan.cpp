// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/atan.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, atan)
{
    Shape shape{3};
    auto p = make_shared<op::Parameter>(element::f32, shape);
    auto atan = make_shared<op::v0::Atan>(p);
    auto fun = make_shared<Function>(OutputVector{atan}, ParameterVector{p});

    std::vector<float> inputs{-0.25f, 0.f, 0.25f};
    std::vector<float> expected_result{-0.24497866f, 0.00000000f, 0.24497866f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(shape, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), shape);
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
