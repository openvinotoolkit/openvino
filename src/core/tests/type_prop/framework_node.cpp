// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;

TEST(type_prop, framework_node) {
    auto param = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 64});
    auto f_node = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{param});
    f_node->set_output_type(0, ov::element::i64, ov::Shape{1, 64});

    // Set partially dynamic shape
    param->set_partial_shape(ov::PartialShape{ov::Dimension::dynamic(), 64});
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set dynamic shape
    param->set_partial_shape(ov::PartialShape::dynamic(2));
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set fully dynamic shape
    param->set_partial_shape(ov::PartialShape::dynamic());
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set original static shape
    param->set_partial_shape(ov::Shape{1, 64});
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape({1, 64}));

    // Set different static shape
    param->set_partial_shape(ov::Shape{2, 64});
    param->validate_and_infer_types();

    ASSERT_THROW(f_node->validate_and_infer_types(), ov::Exception);
}

TEST(type_prop, dynamic_framework_node_with_dynamic_input) {
    auto param = std::make_shared<ov::opset8::Parameter>(ov::element::dynamic, ov::PartialShape::dynamic());
    auto f_node = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{param});

    // Set partially dynamic shape
    param->set_partial_shape(ov::PartialShape{ov::Dimension::dynamic(), 64});
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set dynamic shape with static rank
    param->set_partial_shape(ov::PartialShape::dynamic(2));
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set static shape
    param->set_partial_shape(ov::PartialShape({1, 64}));
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set static type
    param->set_element_type(ov::element::f32);
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_element_type(0), ov::element::dynamic);
}

TEST(type_prop, dynamic_framework_node_with_static_rank) {
    auto param = std::make_shared<ov::opset8::Parameter>(ov::element::dynamic, ov::PartialShape::dynamic(2));
    auto f_node = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{param});

    // Set partially dynamic shape
    param->set_partial_shape(ov::PartialShape{ov::Dimension::dynamic(), 64});
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());

    // Set static shape
    param->set_partial_shape(ov::PartialShape({1, 64}));
    param->validate_and_infer_types();

    OV_ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), ov::PartialShape::dynamic());
}
