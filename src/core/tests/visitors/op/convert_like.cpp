// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, convert_like_op) {
    NodeBuilder::get_ops().register_factory<opset1::ConvertLike>();
    auto data = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto like = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});

    auto convertLike = make_shared<opset1::ConvertLike>(data, like);
    NodeBuilder builder(convertLike, {data, like});
    auto g_convertLike = ov::as_type_ptr<opset1::Concat>(builder.create());

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
