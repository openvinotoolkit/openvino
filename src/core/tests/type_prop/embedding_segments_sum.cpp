// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embedding_segments_sum.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, ess_default_ctor) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 6});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto op = make_shared<op::v3::EmbeddingSegmentsSum>();
    op->set_arguments(OutputVector{emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 2, 6}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop, ess) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                         indices,
                                                         segment_ids,
                                                         num_segments,
                                                         default_index,
                                                         per_sample_weights);
    EXPECT_TRUE(ess->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
    EXPECT_TRUE(indices->get_partial_shape().same_scheme(per_sample_weights->get_partial_shape()));
    EXPECT_EQ(ess->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(segment_ids->get_partial_shape().rank().get_length(), 1);
}

TEST(type_prop, ess_dynamic_emb_table_number_segment) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                         indices,
                                                         segment_ids,
                                                         num_segments,
                                                         default_index,
                                                         per_sample_weights);

    EXPECT_TRUE(ess->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, ess_fail_indices_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid indices type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES type must be i32 or i64"));
    } catch (...) {
        FAIL() << "INDICES type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_segment_ids_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid segment_ids type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("SEGMENT_IDS type must be i32 or i64"));
    } catch (...) {
        FAIL() << "SEGMENT_IDS type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_number_segments_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid segment_ids type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("NUM_SEGMENTS type must be i32 or i64"));
    } catch (...) {
        FAIL() << "NUM_SEGMENTS type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_default_index_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid default_index type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("DEFAULT_INDEX type must be i32 or i64"));
    } catch (...) {
        FAIL() << "DEFAULT_INDEX type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Segment_ids element type (i64) must match indices element type (i32)"));
    } catch (...) {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type_1) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Default_index element type (i32) must match indices element type (i64)"));
    } catch (...) {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type_2) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Per sample weight element type (i64) must "
                                         "match embedding table element type (f32)"));
    } catch (...) {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type_3) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Num_segments element type (i32) must match Segment_ids element type (i64)"));
    } catch (...) {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_shape) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES and PER_SAMPLE_WEIGHTS shape must be same"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_num_segments_scalar) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("NUM_SEGMENTS must be a scalar"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_default_index_scalar) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("DEFAULT_INDEX must be a scalar"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_indices_1d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4, 2});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES must be 1D"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_segment_ids_1d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 2});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("SEGMENT_IDS must be 1D"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_per_sample_weights_1d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 2});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                             indices,
                                                             segment_ids,
                                                             num_segments,
                                                             default_index,
                                                             per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("PER_SAMPLE_WEIGHTS must be 1D"));
    } catch (...) {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_emb_table_0d) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    OV_EXPECT_THROW(auto op = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                                        indices,
                                                                        segment_ids,
                                                                        num_segments,
                                                                        default_index,
                                                                        per_sample_weights),
                    NodeValidationFailure,
                    HasSubstr("EMB_TABLE can't be a scalar"));
}

TEST(type_prop, ess_4_args_api) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table, indices, segment_ids, num_segments);
    EXPECT_TRUE(ess->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(ess->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(segment_ids->get_partial_shape().rank().get_length(), 1);
}

TEST(type_prop, ess_fail_indices_element_type_4_args_api) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    try {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table, indices, segment_ids, num_segments);
        FAIL() << "Invalid indices type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES type must be i32 or i64"));
    } catch (...) {
        FAIL() << "INDICES type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_num_segment_const) {
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = ov::op::v0::Constant::create(element::i64, Shape{}, {3});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table, indices, segment_ids, num_segments);
    EXPECT_TRUE(ess->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(ess->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(segment_ids->get_partial_shape().rank().get_length(), 1);
}

TEST(type_prop, ess_num_segment_interval_label_propagation) {
    auto emb_shape = PartialShape{{5, 10}, {2, 4}, {1, 3}};
    auto emb_symbols = set_shape_symbols(emb_shape);
    auto num_segm_shape = PartialShape{{6, 8}};
    auto num_segm_symbols = set_shape_symbols(num_segm_shape);

    auto num_segments = make_shared<ov::op::v0::Parameter>(element::i64, num_segm_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(num_segments);
    auto num_segm_squeeze = make_shared<op::v0::Squeeze>(shape_of);

    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, emb_shape);
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    auto op = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table, indices, segment_ids, num_segm_squeeze);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{6, 8}, {2, 4}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(num_segm_symbols[0], emb_symbols[1], emb_symbols[2]));
}
