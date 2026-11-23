// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_prepare.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::v0::Parameter, ov::op::v17::QSAPrepare, ov::test::NodeBuilder;

TEST(attributes, qsa_prepare_v17) {
    NodeBuilder::opset().insert<QSAPrepare>();
    const auto raw_index_keys = std::make_shared<Parameter>(element::f32, Shape{1, 8, 128});
    const auto past_block_index_keys = std::make_shared<Parameter>(element::f32, Shape{1, 2, 128});
    const auto key_length = std::make_shared<Parameter>(element::i32, Shape{1});
    const auto rope_cos = std::make_shared<Parameter>(element::f32, Shape{1, 8, 64});
    const auto rope_sin = std::make_shared<Parameter>(element::f32, Shape{1, 8, 64});
    const auto k_norm_weight = std::make_shared<Parameter>(element::f32, Shape{128});
    const auto past_incomplete_raw_keys = std::make_shared<Parameter>(element::f32, Shape{1, 3, 128});
    const auto past_incomplete_rope = std::make_shared<Parameter>(element::f32, Shape{1, 128});

    const auto op = std::make_shared<QSAPrepare>(raw_index_keys,
                                                 past_block_index_keys,
                                                 key_length,
                                                 rope_cos,
                                                 rope_sin,
                                                 k_norm_weight,
                                                 past_incomplete_raw_keys,
                                                 past_incomplete_rope,
                                                 4,
                                                 1e-5f);
    NodeBuilder builder(op,
                        {raw_index_keys,
                         past_block_index_keys,
                         key_length,
                         rope_cos,
                         rope_sin,
                         k_norm_weight,
                         past_incomplete_raw_keys,
                         past_incomplete_rope});
    const auto g_op = ov::as_type_ptr<QSAPrepare>(builder.create());

    constexpr auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_key_block_size(), op->get_key_block_size());
    EXPECT_EQ(g_op->get_epsilon(), op->get_epsilon());
}
}  // namespace ov::test
