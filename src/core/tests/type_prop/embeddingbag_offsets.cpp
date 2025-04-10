// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_offsets.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace ov;
using namespace testing;

TEST(type_prop, embedding_bag_offsets_default_ctor) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 6});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto op = std::make_shared<op::v15::EmbeddingBagOffsets>();
    op->set_arguments(OutputVector{emb_table, indices, offsets, default_index, per_sample_weights});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2, 6}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_reduction_mean) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto reduction = op::v15::EmbeddingBagOffsets::Reduction::MEAN;

    auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, reduction);
    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(embedding_bag_offsets->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(offsets->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::MEAN);
}

TEST(type_prop, embedding_bag_reduction_mean_index) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto reduction = op::v15::EmbeddingBagOffsets::Reduction::MEAN;

    auto embedding_bag_offsets =
        std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, default_index, reduction);
    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(embedding_bag_offsets->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(offsets->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::MEAN);
}

TEST(type_prop, embedding_bag_offsets_labeled_interval_dims) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto off_shape = PartialShape{{6, 8}};
    auto off_symbols = set_shape_symbols(off_shape);

    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{{3, 4}});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, off_shape);
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{3, 4}});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto op =
        std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, default_index, per_sample_weights);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{6, 8}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(off_symbols[0], emb_symbols[1], emb_symbols[2]));
    EXPECT_EQ(op->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_offsets) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto embedding_bag_offsets =
        std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, default_index, per_sample_weights);
    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_TRUE(indices->get_partial_shape().same_scheme(per_sample_weights->get_partial_shape()));
    EXPECT_EQ(embedding_bag_offsets->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(offsets->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_offsets_dynamic_emb_table) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto embedding_bag_offsets =
        std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, default_index, per_sample_weights);

    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(PartialShape{3, Dimension::dynamic()}));
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_offsets_dynamic_offsets) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto embedding_bag_offsets =
        std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, default_index, per_sample_weights);

    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_offsets_dynamic_emb_table_offsets) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto reduction = op::v15::EmbeddingBagOffsets::Reduction::SUM;

    auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                indices,
                                                                                offsets,
                                                                                default_index,
                                                                                per_sample_weights,
                                                                                reduction);

    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_offsets_fail_indices_element_type) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("INDICES type must be i32 or i64"));
}

TEST(type_prop, embedding_bag_offsets_fail_offsets_element_type) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("OFFSETS type must be i32 or i64"));
}

TEST(type_prop, embedding_bag_offsets_fail_default_index_element_type) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("DEFAULT_INDEX type must be i32 or i64"));
}

TEST(type_prop, embedding_bag_offsets_fail_mismatch_element_type) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("Offsets element type (i64) must match indices element type (i32)"));
}

TEST(type_prop, embedding_bag_offsets_fail_mismatch_element_type_1) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("Default_index element type (i32) must match indices element type (i64)"));
}

TEST(type_prop, embedding_bag_offsets_fail_mismatch_element_type_2) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("Per sample weight element type (i64) must "
                              "match embedding table element type (f32)"));
}

TEST(type_prop, embedding_bag_offsets_fail_mismatch_shape) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("INDICES and PER_SAMPLE_WEIGHTS shape must be same"));
}

TEST(type_prop, embedding_bag_offsets_fail_default_index_scalar) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("DEFAULT_INDEX must be a scalar"));
}

TEST(type_prop, embedding_bag_offsets_fail_indices_1d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4, 2});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("INDICES must be 1D"));
}

TEST(type_prop, embedding_bag_offsets_fail_emb_table_0d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto op = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                             indices,
                                                                             offsets,
                                                                             default_index,
                                                                             per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("EMB_TABLE can't be a scalar"));
}

TEST(type_prop, embedding_bag_offsets_fail_offsets_1d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 2});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("OFFSETS must be 1D"));
}

TEST(type_prop, embedding_bag_offsets_fail_per_sample_weights_1d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 2});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("PER_SAMPLE_WEIGHTS must be 1D"));
}

TEST(type_prop, embedding_bag_offsets_fail_per_sample_weights_reduction_mean) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 2});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto reduction = op::v15::EmbeddingBagOffsets::Reduction::MEAN;

    OV_EXPECT_THROW(auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table,
                                                                                                indices,
                                                                                                offsets,
                                                                                                default_index,
                                                                                                per_sample_weights,
                                                                                                reduction),
                    NodeValidationFailure,
                    HasSubstr("Per sample weights can only be used in Reduction::SUM mode"));
}

TEST(type_prop, embedding_bag_offsets_3_args_api) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});

    auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets);
    EXPECT_TRUE(embedding_bag_offsets->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(embedding_bag_offsets->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(offsets->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(embedding_bag_offsets->get_reduction(), op::v15::EmbeddingBagOffsets::Reduction::SUM);
}

TEST(type_prop, embedding_bag_offsets_fail_indices_element_type_3_args_api) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});

    OV_EXPECT_THROW(
        auto embedding_bag_offsets = std::make_shared<op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets),
        NodeValidationFailure,
        HasSubstr("INDICES type must be i32 or i64"));
}
