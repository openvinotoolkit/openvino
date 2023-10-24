// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, result) {
    const auto arg_shape = Shape{1, 2, 3, 4, 5};
    auto arg = make_shared<ov::op::v0::Constant>(element::f32, arg_shape);

    auto result = make_shared<ov::op::v0::Result>(arg);

    EXPECT_EQ(result->get_output_element_type(0), element::f32);
    EXPECT_EQ(result->get_output_partial_shape(0).to_shape(), arg_shape);
}

TEST(type_prop, result_dynamic_shape) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto result = make_shared<ov::op::v0::Result>(arg);

    EXPECT_EQ(result->get_output_element_type(0), element::f32);
    EXPECT_TRUE(result->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, result_layout) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto result = make_shared<ov::op::v0::Result>(a);
    result->set_layout("NHWC");
    EXPECT_EQ(result->get_layout(), "NHWC");
    result->set_layout(ov::Layout());
    EXPECT_TRUE(result->get_layout().empty());
    EXPECT_EQ(result->output(0).get_rt_info().count(ov::LayoutAttribute::get_type_info_static()), 0);
}

TEST(type_prop, result_layout_empty) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto result = make_shared<ov::op::v0::Result>(a);
    EXPECT_TRUE(result->get_layout().empty());
}

TEST(type_prop, result_layout_invalid) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto result = make_shared<ov::op::v0::Result>(a);
    result->output(0).get_rt_info()[ov::LayoutAttribute::get_type_info_static()] = "NCHW";  // incorrect way
    ASSERT_THROW(result->get_layout(), ov::Exception);
}
