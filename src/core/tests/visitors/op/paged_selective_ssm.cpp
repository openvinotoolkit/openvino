// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_selective_ssm.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {

TEST(attributes, paged_selective_ssm_default_attrs) {
    NodeBuilder::opset().insert<op::internal::PagedSelectiveSSM>();
    const auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4});
    const auto dt = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 2, 8});
    const auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto C = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 2, 8});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8, 8});
    const auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = std::make_shared<op::internal::PagedSelectiveSSM>(
        OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval});

    NodeBuilder builder(op, {A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval});
    const auto g_op = as_type_ptr<op::internal::PagedSelectiveSSM>(builder.create());

    EXPECT_EQ(builder.get_value_map_size(), 0);
    EXPECT_NE(g_op, nullptr);
}

}  // namespace ov::test
