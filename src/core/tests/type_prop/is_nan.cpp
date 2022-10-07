// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, isnan_output_shape) {
    const auto data = make_shared<opset10::Parameter>(ov::element::f16, Shape{4, 2});
    const auto isnan = make_shared<opset10::IsNaN>(data);

    EXPECT_TRUE(isnan->get_output_partial_shape(0).same_scheme(PartialShape{4, 2}))
        << "The output shape of IsNaN is incorrect";
    ASSERT_EQ(isnan->get_shape(), (Shape{4, 2})) << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_sample_dynamic_batch) {
    const auto data = make_shared<opset10::Parameter>(ov::element::f16, PartialShape{Dimension::dynamic(), 21, 37});
    const auto isnan = make_shared<opset10::IsNaN>(data);

    EXPECT_TRUE(isnan->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 21, 37}))
        << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_output_type) {
    const auto data = make_shared<opset10::Parameter>(ov::element::f16, Shape{4, 2});
    const auto isnan = make_shared<opset10::IsNaN>(data);

    EXPECT_EQ(isnan->get_element_type(), ov::element::boolean) << "The output element type of IsNaN is not a boolean.";
}
