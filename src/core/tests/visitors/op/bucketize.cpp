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

TEST(attributes, bucketize_v3_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset3::Bucketize>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    auto bucketize = make_shared<opset3::Bucketize>(data, buckets);
    NodeBuilder builder(bucketize);

    auto g_bucketize = ov::as_type_ptr<opset3::Bucketize>(builder.create());

    EXPECT_EQ(g_bucketize->get_output_type(), bucketize->get_output_type());
    EXPECT_EQ(g_bucketize->get_with_right_bound(), bucketize->get_with_right_bound());
}

TEST(attributes, bucketize_v3_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset3::Bucketize>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    element::Type output_type = element::i32;
    bool with_right_bound = false;

    auto bucketize = make_shared<opset3::Bucketize>(data, buckets, output_type, with_right_bound);
    NodeBuilder builder(bucketize);

    auto g_bucketize = ov::as_type_ptr<opset3::Bucketize>(builder.create());

    EXPECT_EQ(g_bucketize->get_output_type(), bucketize->get_output_type());
    EXPECT_EQ(g_bucketize->get_with_right_bound(), bucketize->get_with_right_bound());
}
