// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;

TEST(type_prop, elu) {
    ov::Shape data_shape{2, 4};
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_shape);
    auto elu = make_shared<ov::op::v0::Elu>(data, 1);
    ASSERT_EQ(elu->get_element_type(), ov::element::f32);
    ASSERT_EQ(elu->get_shape(), data_shape);
}
