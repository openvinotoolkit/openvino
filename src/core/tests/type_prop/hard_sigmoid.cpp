// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, hardsigmoid) {
    const Shape data_shape{3, 5};

    const auto P = make_shared<op::Parameter>(element::f32, data_shape);
    const auto alpha = op::Constant::create<float>(P->get_element_type(), Shape{}, {0.1f});
    const auto beta = op::Constant::create<float>(P->get_element_type(), Shape{}, {1.2f});
    const auto H = make_shared<op::HardSigmoid>(P, alpha, beta);
    ASSERT_EQ(H->get_element_type(), element::f32);
    ASSERT_EQ(H->get_shape(), data_shape);
}
