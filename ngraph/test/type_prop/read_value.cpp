// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/read_value.hpp"

#include "gtest/gtest.h"
#include "ngraph/op/parameter.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, read_value_deduce) {
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<op::v3::ReadValue>(input, "variable_id");

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_shape(), (Shape{1, 2, 64, 64}));
}
