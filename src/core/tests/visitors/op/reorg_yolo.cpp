// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, reorg_yolo_op_stride) {
    NodeBuilder::get_ops().register_factory<opset3::ReorgYolo>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 64, 26, 26});

    const auto op = make_shared<op::v0::ReorgYolo>(data, 2);
    NodeBuilder builder(op);
    const auto g_op = ov::as_type_ptr<op::v0::ReorgYolo>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
}

TEST(attributes, reorg_yolo_op_strides) {
    NodeBuilder::get_ops().register_factory<opset3::ReorgYolo>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 64, 26, 26});

    const auto op = make_shared<op::v0::ReorgYolo>(data, Strides{2});
    NodeBuilder builder(op);
    const auto g_op = ov::as_type_ptr<op::v0::ReorgYolo>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
}
