// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

TEST(type_prop, is_inf_default) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<IsInf>(data);

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_dynamic_batch) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 64, 256, 256});
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_scalar) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{1});
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({1}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_bfloat16) {
    const auto data = make_shared<Parameter>(element::bf16, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_float16) {
    const auto data = make_shared<Parameter>(element::f16, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_float64) {
    const auto data = make_shared<Parameter>(element::f64, PartialShape{1, 64, 256, 256});
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({1, 64, 256, 256}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_interval) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{Dimension(2, 4), Dimension(-1, 5)});
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape({Dimension(2, 4), Dimension(-1, 5)}))
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_dynamic) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto is_inf = make_shared<IsInf>(data, IsInf::Attributes{});

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape::dynamic())
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_negative) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    IsInf::Attributes attributes{};
    attributes.detect_positive = false;
    const auto is_inf = make_shared<IsInf>(data, attributes);

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape::dynamic())
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_positive) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    IsInf::Attributes attributes{};
    attributes.detect_negative = false;
    const auto is_inf = make_shared<IsInf>(data, attributes);

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape::dynamic())
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_all) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    IsInf::Attributes attributes{};
    attributes.detect_positive = true;
    attributes.detect_negative = true;
    const auto is_inf = make_shared<IsInf>(data, attributes);

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape::dynamic())
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}

TEST(type_prop, is_inf_none) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    IsInf::Attributes attributes{};
    attributes.detect_positive = false;
    attributes.detect_negative = false;
    const auto is_inf = make_shared<IsInf>(data, attributes);

    EXPECT_EQ(is_inf->get_element_type(), element::boolean)
        << "The output element type of IsInf should always be boolean";
    EXPECT_EQ(is_inf->get_output_partial_shape(0), PartialShape::dynamic())
        << "The output shape of IsInf is incorrect, it should be the same as shape of input data";
}
