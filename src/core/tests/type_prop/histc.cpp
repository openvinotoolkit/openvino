// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using namespace testing;

class TypePropHistcV17Test : public TypePropOpTest<op::v17::Histc> {};

TEST_F(TypePropHistcV17Test, default_ctor) {
    auto data = std::make_shared<Parameter>(element::f32, Shape{2, 3});
    auto histc = make_op(data);

    EXPECT_EQ(histc->get_output_element_type(0), element::f32);
    EXPECT_EQ(histc->get_output_partial_shape(0), PartialShape{100});
}

TEST_F(TypePropHistcV17Test, custom_attributes) {
    auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    auto histc = make_op(data, 7, 1.5, 9.0);

    EXPECT_EQ(histc->get_output_element_type(0), element::f16);
    EXPECT_EQ(histc->get_output_partial_shape(0), PartialShape{7});
    EXPECT_EQ(histc->get_bins(), 7);
    EXPECT_DOUBLE_EQ(histc->get_min_val(), 1.5);
    EXPECT_DOUBLE_EQ(histc->get_max_val(), 9.0);
}

TEST_F(TypePropHistcV17Test, multi_dimensional_input_supported) {
    auto data = std::make_shared<Parameter>(element::f64, PartialShape{2, 3, 4});
    auto histc = make_op(data, 11, 0.0, 0.0);

    EXPECT_EQ(histc->get_output_element_type(0), element::f64);
    EXPECT_EQ(histc->get_output_partial_shape(0), PartialShape{11});
}

TEST_F(TypePropHistcV17Test, empty_constant_input) {
    auto data = std::make_shared<Constant>(element::f32, Shape{0}, std::vector<float>{});
    auto histc = make_op(data, 5, 0.0, 0.0);

    EXPECT_EQ(histc->get_output_partial_shape(0), PartialShape{5});
}

TEST_F(TypePropHistcV17Test, attribute_setters) {
    auto data = std::make_shared<Parameter>(element::bf16, Shape{6});
    auto histc = make_op(data, 3, -1.0, 1.0);

    histc->set_bins(9);
    histc->set_min_val(2.0);
    histc->set_max_val(8.0);

    EXPECT_EQ(histc->get_bins(), 9);
    EXPECT_DOUBLE_EQ(histc->get_min_val(), 2.0);
    EXPECT_DOUBLE_EQ(histc->get_max_val(), 8.0);
}

TEST_F(TypePropHistcV17Test, invalid_data_type) {
    auto data = std::make_shared<Parameter>(element::i32, Shape{5});
    OV_EXPECT_THROW(make_op(data), NodeValidationFailure, HasSubstr("floating-point element type"));
}

TEST_F(TypePropHistcV17Test, invalid_negative_bins) {
    auto data = std::make_shared<Parameter>(element::f32, Shape{5});
    OV_EXPECT_THROW(make_op(data, -1, 0.0, 0.0), NodeValidationFailure, HasSubstr("bins must be non-negative"));
}

TEST_F(TypePropHistcV17Test, invalid_range) {
    auto data = std::make_shared<Parameter>(element::f32, Shape{5});
    OV_EXPECT_THROW(make_op(data, 5, 2.0, 1.0), NodeValidationFailure, HasSubstr("max_val must be greater than or equal"));
}
