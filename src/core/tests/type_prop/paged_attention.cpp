// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace testing;

TEST(type_prop, paged_attention_static_13_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, Shape{3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, Shape{3, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, Shape{3, 4});
    const auto key_cache = std::make_shared<opset13::Parameter>(element::f32, Shape{6, 2, 5, 4});
    const auto value_cache = std::make_shared<opset13::Parameter>(element::f32, Shape{6, 2, 5, 4});
    const auto past_lens = std::make_shared<opset13::Parameter>(element::i32, Shape{5});
    const auto subsequence_begins = std::make_shared<opset13::Parameter>(element::i32, Shape{5});
    const auto block_indices = std::make_shared<opset13::Parameter>(element::i32, Shape{15});
    const auto block_indices_begins = std::make_shared<opset13::Parameter>(element::i32, Shape{8});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, Shape{});
    const auto sliding_window = std::make_shared<opset13::Parameter>(element::i32, Shape{});
    const auto alibi_slopes = std::make_shared<opset13::Parameter>(element::f32, Shape{9});
    const auto max_context_len = std::make_shared<opset13::Parameter>(element::i32, Shape{});

    ov::OutputVector args = {query,
                             key,
                             value,
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             scale,
                             sliding_window,
                             alibi_slopes,
                             max_context_len};
    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (Shape{3, 4}));
}

TEST(type_prop, paged_attention_static_15_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, Shape{3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, Shape{3, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, Shape{3, 4});
    const auto key_cache = std::make_shared<opset13::Parameter>(element::f32, Shape{6, 2, 5, 4});
    const auto value_cache = std::make_shared<opset13::Parameter>(element::f32, Shape{6, 2, 5, 4});
    const auto past_lens = std::make_shared<opset13::Parameter>(element::i32, Shape{5});
    const auto subsequence_begins = std::make_shared<opset13::Parameter>(element::i32, Shape{5});
    const auto block_indices = std::make_shared<opset13::Parameter>(element::i32, Shape{15});
    const auto block_indices_begins = std::make_shared<opset13::Parameter>(element::i32, Shape{8});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, Shape{});
    const auto sliding_window = std::make_shared<opset13::Parameter>(element::i32, Shape{});
    const auto alibi_slopes = std::make_shared<opset13::Parameter>(element::f32, Shape{9});
    const auto max_context_len = std::make_shared<opset13::Parameter>(element::i32, Shape{});

    const auto rotation_coefficients = std::make_shared<opset13::Parameter>(element::f32, Shape{12});
    const auto rotated_block_indices = std::make_shared<opset13::Parameter>(element::i32, Shape{3});

    ov::OutputVector args = {query,
                             key,
                             value,
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             scale,
                             sliding_window,
                             alibi_slopes,
                             max_context_len,
                             rotation_coefficients,
                             rotated_block_indices};

    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
}
