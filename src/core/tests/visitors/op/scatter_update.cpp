// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_update.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, scatter_update_op) {
    NodeBuilder::opset().insert<ov::op::v3::ScatterUpdate>();
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<ov::op::v0::Parameter>(element::i8, ref_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i16, indices_shape);
    auto U = make_shared<ov::op::v0::Parameter>(element::i8, updates_shape);
    auto A = ov::op::v0::Constant::create(element::i16, Shape{}, {1});
    auto op = make_shared<ov::op::v3::ScatterUpdate>(R, I, U, A);

    NodeBuilder builder(op, {R, I, U, A});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(builder.create()));

    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
