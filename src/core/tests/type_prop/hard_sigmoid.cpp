// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hard_sigmoid.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, hardsigmoid) {
    const Shape data_shape{3, 5};

    const auto P = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto alpha = op::v0::Constant::create<float>(P->get_element_type(), Shape{}, {0.1f});
    const auto beta = op::v0::Constant::create<float>(P->get_element_type(), Shape{}, {1.2f});
    const auto H = make_shared<op::v0::HardSigmoid>(P, alpha, beta);
    ASSERT_EQ(H->get_element_type(), element::f32);
    ASSERT_EQ(H->get_shape(), data_shape);
}
