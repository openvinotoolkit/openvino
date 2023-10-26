// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/read_value.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, read_value_deduce) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<ov::op::v3::ReadValue>(input, "variable_id");

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_shape(), (Shape{1, 2, 64, 64}));
}
