// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packed.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace ov;
using namespace testing;

TEST(type_prop, embedding_bag_packed_default_ctor) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 6});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    auto op = std::make_shared<op::v15::EmbeddingBagPacked>();
    op->set_arguments(OutputVector{emb_table, indices, per_sample_weights});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2, 6}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed_labeled_interval_dims_2in) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto ind_shape = PartialShape{{6, 8}, 4};
    auto ind_symbols = set_shape_symbols(ind_shape);

    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);

    auto op = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{6, 8}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(ind_symbols[0], emb_symbols[1], emb_symbols[2]));
    EXPECT_EQ(op->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed_labeled_interval_dims_2in_reduction) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto ind_shape = PartialShape{{6, 8}, 4};
    auto ind_symbols = set_shape_symbols(ind_shape);
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);
    auto reduction = op::v15::EmbeddingBagPacked::Reduction::MEAN;

    auto op = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, reduction);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{6, 8}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(ind_symbols[0], emb_symbols[1], emb_symbols[2]));
    EXPECT_EQ(op->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::MEAN);
}

TEST(type_prop, embedding_bag_packed_labeled_interval_dims_3in) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto ind_shape = PartialShape{{2, 6}, 4};
    auto ind_symbols = set_shape_symbols(ind_shape);
    auto sample_shape = PartialShape{{4, 8}, 4};
    auto sample_symbols = set_shape_symbols(sample_shape);

    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, sample_shape);

    auto op = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 6}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(ind_symbols[0], emb_symbols[1], emb_symbols[2]));
    EXPECT_EQ(op->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    auto embedding_bag_packed = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights);
    EXPECT_TRUE(embedding_bag_packed->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_TRUE(indices->get_partial_shape().same_scheme(per_sample_weights->get_partial_shape()));
    EXPECT_EQ(embedding_bag_packed->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 2);
    EXPECT_EQ(embedding_bag_packed->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed_dynamic_emb_table) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto default_index = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto embedding_bag_packed = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(embedding_bag_packed->get_output_partial_shape(0).same_scheme(PartialShape{3, Dimension::dynamic()}));
    EXPECT_EQ(embedding_bag_packed->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed_dynamic_indices) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 4});
    auto per_sample_weights =
        std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});

    auto embedding_bag_packed = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(embedding_bag_packed->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(embedding_bag_packed->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed_dynamic_emb_table_indices) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 4});
    auto per_sample_weights =
        std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto reduction = op::v15::EmbeddingBagPacked::Reduction::SUM;
    auto embedding_bag_packed =
        std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights, reduction);

    EXPECT_TRUE(embedding_bag_packed->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(embedding_bag_packed->get_reduction(), op::v15::EmbeddingBagPacked::Reduction::SUM);
}

TEST(type_prop, embedding_bag_packed_fail_indices_element_type) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    OV_EXPECT_THROW(auto embedding_bag_packed =
                        std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("INDICES type must be i32 or i64"));
}

TEST(type_prop, embedding_bag_packed_fail_mismatch_element_type) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});

    OV_EXPECT_THROW(auto embedding_bag_packed =
                        std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("Per sample weight element type (i64) must "
                              "match embedding table element type (f32)"));
}

TEST(type_prop, embedding_bag_packed_fail_mismatch_shape) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 3});

    OV_EXPECT_THROW(auto embedding_bag_packed =
                        std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("INDICES and PER_SAMPLE_WEIGHTS shape must be same"));
}

TEST(type_prop, embedding_bag_packed_fail_indices_1d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    OV_EXPECT_THROW(auto embedding_bag_packed =
                        std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("INDICES must be 2D"));
}

TEST(type_prop, embedding_bag_packed_fail_emb_table_0d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    OV_EXPECT_THROW(auto op = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("EMB_TABLE can't be a scalar"));
}

TEST(type_prop, embedding_bag_packed_fail_per_sample_weights_1d) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    OV_EXPECT_THROW(auto embedding_bag_packed =
                        std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("PER_SAMPLE_WEIGHTS must be 2D"));
}

TEST(type_prop, embedding_bag_packed_fail_per_sample_weights_reduction_mean) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto reduction = op::v15::EmbeddingBagPacked::Reduction::MEAN;
    OV_EXPECT_THROW(
        auto embedding_bag_packed =
            std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights, reduction),
        NodeValidationFailure,
        HasSubstr("Per sample weights can only be used in Reduction::SUM mode."));
}

TEST(type_prop, embedding_bag_packed_2_args_api) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});

    auto embedding_bag_packed = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices);
    EXPECT_TRUE(embedding_bag_packed->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(embedding_bag_packed->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 2);
}

TEST(type_prop, embedding_bag_packed_fail_indices_element_type_2_args_api) {
    auto emb_table = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    OV_EXPECT_THROW(auto embedding_bag_packed = std::make_shared<op::v15::EmbeddingBagPacked>(emb_table, indices),
                    NodeValidationFailure,
                    HasSubstr("INDICES type must be i32 or i64"));
}
