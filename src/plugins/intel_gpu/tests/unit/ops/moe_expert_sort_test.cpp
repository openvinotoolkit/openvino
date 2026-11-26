// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/op/moe_expert_sort.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"

using ov::intel_gpu::op::MoEExpertSort;

namespace ov::test::intel_gpu {

namespace {

std::shared_ptr<MoEExpertSort> make_sort(const ov::element::Type& type, const ov::PartialShape& shape, size_t num_experts) {
    return std::make_shared<MoEExpertSort>(std::make_shared<ov::op::v0::Parameter>(type, shape), num_experts);
}

}  // namespace

TEST(MoEExpertSortOp, infers_output_types_and_shapes) {
    auto sort = make_sort(ov::element::i64, {5, 3}, 8);
    ASSERT_EQ(sort->get_output_size(), 3u);
    EXPECT_EQ(sort->get_output_element_type(0), ov::element::i32);
    EXPECT_EQ(sort->get_output_partial_shape(0), (ov::PartialShape{15}));
    EXPECT_EQ(sort->get_output_partial_shape(1), (ov::PartialShape{8}));
    EXPECT_EQ(sort->get_output_partial_shape(2), (ov::PartialShape{15}));
}

TEST(MoEExpertSortOp, dynamic_tokens_give_dynamic_rows) {
    auto sort = make_sort(ov::element::i32, {-1, 6}, 64);
    EXPECT_EQ(sort->get_output_partial_shape(0), (ov::PartialShape{-1}));
    EXPECT_EQ(sort->get_output_partial_shape(1), (ov::PartialShape{64}));
    EXPECT_EQ(sort->get_output_partial_shape(2), (ov::PartialShape{-1}));
}

TEST(MoEExpertSortOp, top_k_equal_to_num_experts_is_valid) {
    EXPECT_NO_THROW(make_sort(ov::element::i32, {-1, 4}, 4));
}

TEST(MoEExpertSortOp, rejects_non_integer_or_unsupported_ids_type) {
    EXPECT_THROW(make_sort(ov::element::f32, {-1, 2}, 8), ov::NodeValidationFailure);
    EXPECT_THROW(make_sort(ov::element::u8, {-1, 2}, 8), ov::NodeValidationFailure);
    EXPECT_THROW(make_sort(ov::element::dynamic, {-1, 2}, 8), ov::NodeValidationFailure);
}

TEST(MoEExpertSortOp, rejects_wrong_rank) {
    EXPECT_THROW(make_sort(ov::element::i32, {2}, 8), ov::NodeValidationFailure);
    EXPECT_THROW(make_sort(ov::element::i32, {1, 4, 2}, 8), ov::NodeValidationFailure);
    EXPECT_THROW(make_sort(ov::element::i32, ov::PartialShape::dynamic(), 8), ov::NodeValidationFailure);
}

TEST(MoEExpertSortOp, rejects_dynamic_or_empty_top_k) {
    EXPECT_THROW(make_sort(ov::element::i32, {4, -1}, 8), ov::NodeValidationFailure);
    EXPECT_THROW(make_sort(ov::element::i32, {4, 0}, 8), ov::NodeValidationFailure);
}

TEST(MoEExpertSortOp, rejects_top_k_above_num_experts) {
    EXPECT_THROW(make_sort(ov::element::i32, {-1, 9}, 8), ov::NodeValidationFailure);
}

TEST(MoEExpertSortOp, clone_keeps_num_experts) {
    auto sort = make_sort(ov::element::i32, {-1, 2}, 8);
    auto clone = ov::as_type_ptr<MoEExpertSort>(sort->clone_with_new_inputs(sort->input_values()));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_num_experts(), 8u);
}

}  // namespace ov::test::intel_gpu
