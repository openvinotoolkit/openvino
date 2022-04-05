// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/mish.hpp"

#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, mish_0D) {
    auto p = make_shared<op::Parameter>(element::f32, Shape{});
    auto mish = make_shared<op::v4::Mish>(p);
    auto fun = make_shared<Function>(OutputVector{mish}, ParameterVector{p});

    std::vector<std::vector<float>> inputs{{-1.0}, {1.0}, {20.0}};
    std::vector<std::vector<float>> expected_result{{-0.303401}, {0.86509835720062256}, {20.0}};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{}, inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), (Shape{}));
        auto result_data = read_vector<float>(result);
        EXPECT_NEAR(result_data[0], expected_result[i][0], 0.000001);
    }
}
