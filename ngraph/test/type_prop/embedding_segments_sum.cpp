//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, ess)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
        emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
    EXPECT_TRUE(
        ess->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
    EXPECT_TRUE(indices->get_partial_shape().same_scheme(per_sample_weights->get_partial_shape()));
    EXPECT_EQ(ess->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(segment_ids->get_partial_shape().rank().get_length(), 1);
}

TEST(type_prop, ess_dynamic_emb_table_number_segment)
{
    auto emb_table =
        make_shared<op::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
        emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);

    EXPECT_TRUE(ess->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, ess_fail_indices_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::f32, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid indices type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES type must be i32 or i64"));
    }
    catch (...)
    {
        FAIL() << "INDICES type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_segment_ids_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::f32, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid segment_ids type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("SEGMENT_IDS type must be i32 or i64"));
    }
    catch (...)
    {
        FAIL() << "SEGMENT_IDS type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_number_segments_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::f32, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid segment_ids type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("NUM_SEGMENTS type must be i32 or i64"));
    }
    catch (...)
    {
        FAIL() << "NUM_SEGMENTS type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_default_index_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid default_index type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("DEFAULT_INDEX type must be i32 or i64"));
    }
    catch (...)
    {
        FAIL() << "DEFAULT_INDEX type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i32, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Segment_ids element type (i64) must match indices element type (i32)"));
    }
    catch (...)
    {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type_1)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Default_index element type (i32) must match indices element type (i64)"));
    }
    catch (...)
    {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type_2)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Per sample weight element type (i64) must "
                                         "match embedding table element type (f32)"));
    }
    catch (...)
    {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_element_type_3)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i32, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Num_segments element type (i32) must match Segment_ids element type (i64)"));
    }
    catch (...)
    {
        FAIL() << "Element type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_mismatch_shape)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{3});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("INDICES and PER_SAMPLE_WEIGHTS shape must be same"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_num_segments_scalar)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{2});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("NUM_SEGMENTS must be a scalar"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_default_index_scalar)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{2});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("DEFAULT_INDEX must be a scalar"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_indices_1d)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4, 2});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES must be 1D"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_segment_ids_1d)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{3, 2});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("SEGMENT_IDS must be 1D"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_fail_per_sample_weights_1d)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("PER_SAMPLE_WEIGHTS must be 1D"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ess_4_args_api)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});

    auto ess =
        make_shared<op::v3::EmbeddingSegmentsSum>(emb_table, indices, segment_ids, num_segments);
    EXPECT_TRUE(
        ess->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(ess->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(segment_ids->get_partial_shape().rank().get_length(), 1);
}

TEST(type_prop, ess_fail_indices_element_type_4_args_api)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::f32, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});

    try
    {
        auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(
            emb_table, indices, segment_ids, num_segments);
        FAIL() << "Invalid indices type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES type must be i32 or i64"));
    }
    catch (...)
    {
        FAIL() << "INDICES type check failed for unexpected reason";
    }
}

TEST(type_prop, ess_num_segment_const)
{
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = opset3::Constant::create(element::i64, Shape{}, {3});

    auto ess =
        make_shared<op::v3::EmbeddingSegmentsSum>(emb_table, indices, segment_ids, num_segments);
    EXPECT_TRUE(ess->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(ess->get_output_element_type(0), element::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 1);
    EXPECT_EQ(segment_ids->get_partial_shape().rank().get_length(), 1);
}
