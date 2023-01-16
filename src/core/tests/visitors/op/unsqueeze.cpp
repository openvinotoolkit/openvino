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

TEST(attributes, unsqueeze_op) {
    using namespace opset1;

    NodeBuilder::get_ops().register_factory<op::v0::Unsqueeze>();

    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes = make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto op = make_shared<op::v0::Unsqueeze>(param, axes);

    NodeBuilder builder(op, {param, axes});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<op::v0::Unsqueeze>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
