// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ov;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, eye_op) {
    NodeBuilder::get_ops().register_factory<op::v9::Eye>();
    auto num_rows = make_shared<op::v0::Constant>(element::i32, Shape{}, 10);

    const auto eye_like = make_shared<op::v9::Eye>(num_rows, element::Type_t::f32);
    NodeBuilder builder(eye_like);
    auto g_eye = ov::as_type_ptr<op::v9::Eye>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_eye->get_out_type(), eye_like->get_out_type());
}
