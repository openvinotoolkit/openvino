// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset7.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, matrix_nms_v8_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset7::MatrixNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = opset7::MatrixNms::BoxEncodingType::CENTER;
    bool sort_result_descending = false;
    element::Type output_type = element::i32;

    auto nms = make_shared<opset7::MatrixNms>(
        boxes, scores, box_encoding, sort_result_descending, output_type);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset7::MatrixNms>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}

TEST(attributes, matrix_nms_v8_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset7::MatrixNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset7::MatrixNms>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset7::MatrixNms>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}
