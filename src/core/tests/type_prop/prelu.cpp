// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, prelu) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto slope = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    Shape prelu_shape{2, 4};
    auto prelu = make_shared<op::v0::PRelu>(param, slope);
    ASSERT_EQ(prelu->get_element_type(), element::f32);
    ASSERT_EQ(prelu->get_shape(), prelu_shape);
}
