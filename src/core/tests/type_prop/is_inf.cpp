// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, is_inf_default) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_TRUE(is_inf->get_output_partial_shape(0).same_scheme(PartialShape{1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_dynamic_batch) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 64, 256, 256});
    const auto is_inf = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_TRUE(is_inf->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_scalar) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{1});
    const auto is_inf = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_TRUE(is_inf->get_output_partial_shape(0).same_scheme(PartialShape{1}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_bfloat16) {
    const auto data = make_shared<opset10::Parameter>(element::bf16, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_TRUE(is_inf->get_output_partial_shape(0).same_scheme(PartialShape{1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_float16) {
    const auto data = make_shared<opset10::Parameter>(element::f16, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_TRUE(is_inf->get_output_partial_shape(0).same_scheme(PartialShape{1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_float64) {
    const auto data = make_shared<opset10::Parameter>(element::f64, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_TRUE(is_inf->get_output_partial_shape(0).same_scheme(PartialShape{1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}
