// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::GatedDeltaNet> make_gdn(const element::Type& et,
                                                      const PartialShape& q,
                                                      const PartialShape& k,
                                                      const PartialShape& v,
                                                      const PartialShape& state,
                                                      const PartialShape& gate,
                                                      const PartialShape& beta,
                                                      int64_t snapshot_count = 1) {
    auto query = std::make_shared<op::v0::Parameter>(et, q);
    auto key = std::make_shared<op::v0::Parameter>(et, k);
    auto value = std::make_shared<op::v0::Parameter>(et, v);
    auto recurrent_state = std::make_shared<op::v0::Parameter>(et, state);
    auto gate_p = std::make_shared<op::v0::Parameter>(et, gate);
    auto beta_p = std::make_shared<op::v0::Parameter>(et, beta);

    return std::make_shared<op::internal::GatedDeltaNet>(
        OutputVector{query, key, value, recurrent_state, gate_p, beta_p},
        false,
        1e-6f,
        1e-6f,
        snapshot_count);
}

}  // namespace

TEST(type_prop, gated_delta_net_static_f32) {
    const auto op = make_gdn(element::f32,
                             Shape{2, 5, 4, 8},
                             Shape{2, 5, 4, 8},
                             Shape{2, 5, 4, 16},
                             Shape{2, 4, 8, 16},
                             Shape{2, 5, 4},
                             Shape{2, 5, 4});

    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 5, 4, 16}));
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 4, 8, 16}));
}

TEST(type_prop, gated_delta_net_snapshots) {
    for (const int64_t count : {2, 4}) {
        const auto op = make_gdn(element::f32,
                                 PartialShape{{1, 4}, -1, 2, 8},
                                 PartialShape{{1, 4}, -1, 2, 8},
                                 PartialShape{{1, 4}, -1, 4, 16},
                                 PartialShape{{1, 4}, 4, 8, 16},
                                 PartialShape{{1, 4}, -1, 4},
                                 PartialShape{{1, 4}, -1, 4},
                                 count);
        EXPECT_EQ(op->get_output_size(), 2);
        EXPECT_EQ(op->get_output_element_type(1), element::f32);
        EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{count, {1, 4}, 4, 8, 16}));
        const auto clone = as_type_ptr<op::internal::GatedDeltaNet>(op->clone_with_new_inputs(op->input_values()));
        ASSERT_NE(clone, nullptr);
        EXPECT_EQ(clone->get_snapshot_count(), count);
        EXPECT_EQ(clone->get_output_partial_shape(1), op->get_output_partial_shape(1));
        op->set_snapshot_count(count + 1);
        op->validate_and_infer_types();
        EXPECT_EQ(op->get_output_partial_shape(1)[0], Dimension(count + 1));
        op->set_snapshot_count(1);
        op->validate_and_infer_types();
        EXPECT_EQ(op->get_output_size(), 2);
        EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{1, 4}, 4, 8, 16}));
    }
}

TEST(type_prop, gated_delta_net_invalid_snapshot_count) {
    for (const int64_t count : {0, -1}) {
        OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                               Shape{1, 2, 2, 8},
                                               Shape{1, 2, 2, 8},
                                               Shape{1, 2, 4, 16},
                                               Shape{1, 4, 8, 16},
                                               Shape{1, 2, 4},
                                               Shape{1, 2, 4},
                                               count),
                        NodeValidationFailure,
                        testing::HasSubstr("snapshot_count must be positive"));
    }
}

TEST(type_prop, gated_delta_net_static_f16) {
    const auto op = make_gdn(element::f16,
                             Shape{2, 5, 4, 8},
                             Shape{2, 5, 4, 8},
                             Shape{2, 5, 4, 16},
                             Shape{2, 4, 8, 16},
                             Shape{2, 5, 4},
                             Shape{2, 5, 4});

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_element_type(1), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 5, 4, 16}));
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 4, 8, 16}));
}

TEST(type_prop, gated_delta_net_partial_shape_infer) {
    const auto op = make_gdn(element::bf16,
                             PartialShape{{1, 4}, -1, {2, 8}, 64},
                             PartialShape{{1, 4}, -1, {2, 8}, 64},
                             PartialShape{{1, 4}, -1, {2, 8}, {32, 128}},
                             PartialShape{{1, 4}, {2, 8}, 64, {32, 128}},
                             PartialShape{{1, 4}, -1, {2, 8}},
                             PartialShape{{1, 4}, -1, {2, 8}});

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_element_type(1), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 4}, -1, {2, 8}, {32, 128}}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{1, 4}, {2, 8}, 64, {32, 128}}));
}

TEST(type_prop, gated_delta_net_invalid_query_rank) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                           Shape{2, 5, 8},
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 8, 16},
                                           Shape{2, 5, 4},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `query` input should be in [4] list"));
}

TEST(type_prop, gated_delta_net_invalid_gate_rank) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 8, 16},
                                           Shape{2, 5, 4, 1},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `gate` input should be in [3] list"));
}

TEST(type_prop, gated_delta_net_invalid_type) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::i32,
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 8, 16},
                                           Shape{2, 5, 4},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("Element type of `query` input should be in"));
}

TEST(type_prop, gated_delta_net_head_num_mismatch_qk) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 6, 8},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 8, 16},
                                           Shape{2, 5, 4},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in query and key should be the same"));
}

TEST(type_prop, gated_delta_net_head_size_mismatch_qk) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 32},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 32, 16},
                                           Shape{2, 5, 4},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("The head size in key and query should be the same"));
}

TEST(type_prop, gated_delta_net_gate_beta_head_num_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 8, 16},
                                           Shape{2, 5, 6},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in gate, beta, and value should be the same"));
}

TEST(type_prop, gated_delta_net_gqa_v_num_heads_greater_than_num_heads) {
    // GQA: query/key have 4 heads, value/gate/beta/state have 8 heads
    const auto op = make_gdn(element::f32,
                             Shape{2, 5, 4, 8},
                             Shape{2, 5, 4, 8},
                             Shape{2, 5, 8, 16},
                             Shape{2, 8, 8, 16},
                             Shape{2, 5, 8},
                             Shape{2, 5, 8});

    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 5, 8, 16}));
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 8, 8, 16}));
}

TEST(type_prop, gated_delta_net_state_shape_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_gdn(element::f32,
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 4, 16},
                                           Shape{2, 4, 8, 32},
                                           Shape{2, 5, 4},
                                           Shape{2, 5, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr(
                        "The dim at shape[-1] of recurrent_state and head size of value should be the same, but got"));
}
}  // namespace ov::test