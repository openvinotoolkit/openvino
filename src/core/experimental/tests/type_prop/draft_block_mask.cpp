// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/draft_block_mask.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::test {

class TypePropDraftBlockMaskTest : public TypePropOpTest<op::experimental::DraftBlockMask> {};

TEST_F(TypePropDraftBlockMaskTest, static_shapes_3_input) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{2, 2});

    const auto op = make_op(q_windows, k_windows, local_mask, /*num_heads=*/2, /*seqlen=*/2, /*topk=*/3);

    EXPECT_EQ(op->get_output_element_type(0), element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 4, 4}));
    EXPECT_EQ(op->get_seqlen(), 2);
    EXPECT_EQ(op->get_topk(), 3);
    EXPECT_FALSE(op->has_evaluate());
}

TEST_F(TypePropDraftBlockMaskTest, u8_local_mask_accepted) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{2, 2});

    const auto op = make_op(q_windows, k_windows, local_mask, /*num_heads=*/2, /*seqlen=*/2, /*topk=*/3);

    EXPECT_EQ(op->get_output_element_type(0), element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 4, 4}));
}

TEST_F(TypePropDraftBlockMaskTest, dynamic_5_input_fully_dynamic_qkv) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2);

    EXPECT_EQ(op->get_output_element_type(0), element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, Dimension::dynamic(), Dimension::dynamic()}));
    // Sentinel: seqlen/topk have no compile-time value in the runtime-input constructor.
    EXPECT_EQ(op->get_seqlen(), -1);
    EXPECT_EQ(op->get_topk(), -1);
    EXPECT_FALSE(op->has_evaluate());
}

TEST_F(TypePropDraftBlockMaskTest, dynamic_5_input_static_head_dim_narrows_output) {
    const auto q_windows =
        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{Dimension::dynamic(), 128, 4});
    const auto k_windows =
        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{Dimension::dynamic(), 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});

    const auto op = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2, /*batch_size=*/1);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropDraftBlockMaskTest, dynamic_5_input_head_dim_not_divisible_throws) {
    const auto q_windows =
        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{Dimension::dynamic(), 128, 5});
    const auto k_windows =
        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{Dimension::dynamic(), 128, 5});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});

    OV_EXPECT_THROW(std::ignore = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2),
                    NodeValidationFailure,
                    testing::HasSubstr("must be divisible by num_heads"));
}

TEST_F(TypePropDraftBlockMaskTest, dynamic_5_input_wrong_seqlen_element_type_throws) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});

    OV_EXPECT_THROW(std::ignore = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2),
                    NodeValidationFailure,
                    testing::HasSubstr("i32 or i64 seqlen/topk"));
}

TEST_F(TypePropDraftBlockMaskTest, dynamic_5_input_wrong_topk_rank_throws) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2, 2});

    OV_EXPECT_THROW(std::ignore = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `topk`"));
}

// Regression test: Nk must be divisible by local_mask's own HW extent -- otherwise a non-divisible
// Nk would silently truncate via integer division, instead of being rejected.
TEST_F(TypePropDraftBlockMaskTest, nk_not_divisible_by_HW_throws) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{5, 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{2, 2});

    OV_EXPECT_THROW(std::ignore = make_op(q_windows, k_windows, local_mask, /*num_heads=*/2, /*seqlen=*/2, /*topk=*/3),
                    NodeValidationFailure,
                    testing::HasSubstr("k_windows block count"));
}

// local_mask must be square: [HW, HW]. A non-square mask must be rejected regardless of Nq/Nk.
TEST_F(TypePropDraftBlockMaskTest, local_mask_not_square_throws) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{2, 3});

    OV_EXPECT_THROW(std::ignore = make_op(q_windows, k_windows, local_mask, /*num_heads=*/2, /*seqlen=*/2, /*topk=*/3),
                    NodeValidationFailure,
                    testing::HasSubstr("local_mask must be square"));
}

// Regression test documenting the corrected contract: Nq != Nk is VALID when Nk is a multiple of
// local_mask's HW, matching a streaming decoder's growing KV cache (the K-side temporal group count
// is independent of seqlen) -- mirrors the GPU unit test's seqlen=2, k_groups=4, spatial_windows=4
// scenario (intel_gpu's draft_block_mask.cpp test, run_shape's third call).
TEST_F(TypePropDraftBlockMaskTest, nq_ne_nk_streaming_cache_shapes_pass) {
    const auto q_windows =
        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{8, 128, 4});  // seqlen=2, HW=4
    const auto k_windows =
        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{16, 128, 4});  // k_groups=4, HW=4
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{4, 4});

    const auto op = make_op(q_windows, k_windows, local_mask, /*num_heads=*/2, /*seqlen=*/2, /*topk=*/3);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 8, 16}));
}

// Nk % HW == 0 is validated even in the 5-input (runtime seqlen/topk) case -- this needs no seqlen
// VALUE, so it is not gated behind the 3-input-only branch the way Nq == seqlen*HW is.
TEST_F(TypePropDraftBlockMaskTest, dynamic_5_input_nk_not_divisible_by_HW_throws) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{5, 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{2, 2});
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});

    OV_EXPECT_THROW(std::ignore = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2),
                    NodeValidationFailure,
                    testing::HasSubstr("k_windows block count"));
}

TEST_F(TypePropDraftBlockMaskTest, clone_with_new_inputs_3_input) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 4});
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{2, 2});
    const auto op = make_op(q_windows, k_windows, local_mask, /*num_heads=*/2, /*seqlen=*/2, /*topk=*/3);

    const auto cloned = op->clone_with_new_inputs(op->input_values());

    EXPECT_EQ(cloned->get_input_size(), 3u);
    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST_F(TypePropDraftBlockMaskTest, clone_with_new_inputs_5_input) {
    const auto q_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto k_windows = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto local_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto topk = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto op = make_op(q_windows, k_windows, local_mask, seqlen, topk, /*num_heads=*/2);

    const auto cloned = op->clone_with_new_inputs(op->input_values());

    EXPECT_EQ(cloned->get_input_size(), 5u);
    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

}  // namespace ov::test
