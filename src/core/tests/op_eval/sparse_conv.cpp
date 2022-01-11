// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/sparse_conv.hpp"

#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, sparse_conv_single_channel) {
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{2, 1});
    auto arg1 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{3, 3, 3, 1, 1});
    auto arg3 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto conv = make_shared<op::v1::SparseConv>(arg0, arg1, arg2, arg3);
    auto fun = make_shared<Function>(OutputVector{conv}, ParameterVector{arg0, arg1, arg2, arg3});

    std::vector<float> features{1.0f, 1.0f};
    std::vector<float> inp_pos{1.46057f, 3.3381f, 0.504631f,
                               1.00087f, 2.48036f, 1.01154f};
    std::vector<float> kernel(3 * 3 * 3);
    std::iota(kernel.begin(), kernel.end(), 1);

    std::vector<float> offset(3, 0.0f);

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{2, 1}, features),
                               make_host_tensor<element::Type_t::f32>(Shape{2, 3}, inp_pos),
                               make_host_tensor<element::Type_t::f32>(Shape{3, 3, 3, 1, 1}, kernel),
                               make_host_tensor<element::Type_t::f32>(Shape{3}, offset)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_data = read_vector<float>(result);
    EXPECT_NEAR(result_data[0], 34.f, 0.000001);
    EXPECT_NEAR(result_data[1], 22.f, 0.000001);
}
