// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packedsum.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, ebps_default_ctor) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 6});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    auto op = make_shared<op::v3::EmbeddingBagPackedSum>();
    op->set_arguments(OutputVector{emb_table, indices, per_sample_weights});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2, 6}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop, ebps_labeled_interval_dims_2in) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto ind_shape = PartialShape{{6, 8}, 4};
    auto ind_symbols = set_shape_symbols(ind_shape);

    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);

    auto op = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{6, 8}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(ind_symbols[0], emb_symbols[1], emb_symbols[2]));
}

TEST(type_prop, ebps_labeled_interval_dims_3in) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto ind_shape = PartialShape{{2, 6}, 4};
    auto ind_symbols = set_shape_symbols(ind_shape);
    auto sample_shape = PartialShape{{4, 8}, 4};
    auto sample_symbols = set_shape_symbols(sample_shape);

    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, sample_shape);

    auto op = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 6}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(ind_symbols[0], emb_symbols[1], emb_symbols[2]));
}

TEST(type_prop, ebps) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_TRUE(indices->get_partial_shape().same_scheme(per_sample_weights->get_partial_shape()));
    EXPECT_EQ(ebps->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 2);
}

TEST(type_prop, ebps_dynamic_emb_table) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(PartialShape{3, Dimension::dynamic()}));
}

TEST(type_prop, ebps_dynamic_indices) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
}

TEST(type_prop, ebps_dynamic_emb_table_indices) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(
        ebps->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, ebps_fail_indices_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    try {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid indices type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES type must be i32 or i64"));
    } catch (...) {
        FAIL() << "INDICES type check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_fail_mismatch_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});

    try {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Per sample weight element type (i64) must "
                                         "match embedding table element type (f32)"));
    } catch (...) {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_fail_mismatch_shape) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 3});

    try {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES and PER_SAMPLE_WEIGHTS shape must be same"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_fail_indices_1d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    try {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES must be 2D"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_fail_emb_table_0d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    OV_EXPECT_THROW(auto op = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("EMB_TABLE can't be a scalar"));
}

TEST(type_prop, ebps_fail_per_sample_weights_1d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    try {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("PER_SAMPLE_WEIGHTS must be 2D"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_2_args_api) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices);
    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(ebps->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 2);
}

TEST(type_prop, ebps_fail_indices_element_type_2_args_api) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    try {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices);
        FAIL() << "Invalid indices type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES type must be i32 or i64"));
    } catch (...) {
        FAIL() << "INDICES type check failed for unexpected reason";
    }
}
