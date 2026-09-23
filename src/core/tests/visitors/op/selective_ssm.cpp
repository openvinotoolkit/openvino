// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {

TEST(attributes, selective_ssm_default_attrs) {
    NodeBuilder::opset().insert<op::internal::SelectiveSSM>();
    const auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4});
    const auto dt = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 4});
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 2, 8});
    const auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 4, 8});
    const auto C = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 2, 8});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8, 8});

    const auto op = std::make_shared<op::internal::SelectiveSSM>(OutputVector{A, dt, B, x, C, state});
    NodeBuilder builder(op, {A, dt, B, x, C, state});
    const auto g_op = as_type_ptr<op::internal::SelectiveSSM>(builder.create());

    EXPECT_EQ(builder.get_value_map_size(), 0);
    EXPECT_NE(g_op, nullptr);
}

}  // namespace ov::test
