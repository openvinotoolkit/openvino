// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/opsets/opset8.hpp>
#include <openvino/op/util/framework_node.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, FrameworkNode) {
    auto param = std::make_shared<ngraph::opset8::Parameter>(element::i64, Shape{1, 64});
    auto f_node = std::make_shared<ov::op::util::FrameworkNode>(OutputVector{param});
    f_node->set_output_type(0, element::i64, Shape{1, 64});

    // Set partially dynamic shape
    param->set_partial_shape(PartialShape{Dimension::dynamic(), 64});
    param->validate_and_infer_types();

    ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), PartialShape::dynamic());

    // Set dynamic shape
    param->set_partial_shape(PartialShape::dynamic(2));
    param->validate_and_infer_types();

    ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), PartialShape::dynamic());

    // Set fully dynamic shape
    param->set_partial_shape(PartialShape::dynamic());
    param->validate_and_infer_types();

    ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), PartialShape::dynamic());

    // Set original static shape
    param->set_partial_shape(Shape{1, 64});
    param->validate_and_infer_types();

    ASSERT_NO_THROW(f_node->validate_and_infer_types());
    ASSERT_EQ(f_node->get_output_partial_shape(0), PartialShape({1, 64}));

    // Set different static shape
    param->set_partial_shape(Shape{2, 64});
    param->validate_and_infer_types();

    ASSERT_THROW(f_node->validate_and_infer_types(), ngraph_error::exception);
}
