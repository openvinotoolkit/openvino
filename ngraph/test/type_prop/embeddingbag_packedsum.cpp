//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, ebps)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_TRUE(indices->get_partial_shape().same_scheme(per_sample_weights->get_partial_shape()));
    EXPECT_EQ(ebps->get_output_element_type(0), element::Type_t::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 2);
}

TEST(type_prop, ebps_dynamic_emb_table)
{
    auto emb_table =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{5, Dimension::dynamic()});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});
    auto default_index = make_shared<op::Parameter>(element::Type_t::i64, Shape{});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(
        ebps->get_output_partial_shape(0).same_scheme(PartialShape{3, Dimension::dynamic()}));
}

TEST(type_prop, ebps_dynamic_indices)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices =
        make_shared<op::Parameter>(element::Type_t::i64, PartialShape{Dimension::dynamic(), 4});
    auto per_sample_weights =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{Dimension::dynamic(), 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(
        ebps->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2}));
}

TEST(type_prop, ebps_dynamic_emb_table_indices)
{
    auto emb_table =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{5, Dimension::dynamic()});
    auto indices =
        make_shared<op::Parameter>(element::Type_t::i64, PartialShape{Dimension::dynamic(), 4});
    auto per_sample_weights =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{Dimension::dynamic(), 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);

    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, ebps_fail_indices_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});

    try
    {
        auto ebps =
            make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
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

TEST(type_prop, ebps_fail_mismatch_element_type)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});

    try
    {
        auto ebps =
            make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
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

TEST(type_prop, ebps_fail_mismatch_shape)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{4, 3});

    try
    {
        auto ebps =
            make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
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

TEST(type_prop, ebps_fail_indices_1d)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});

    try
    {
        auto ebps =
            make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("INDICES must be 2D"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_fail_per_sample_weights_1d)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{4});

    try
    {
        auto ebps =
            make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
        FAIL() << "Invalid mismatch of shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("PER_SAMPLE_WEIGHTS must be 2D"));
    }
    catch (...)
    {
        FAIL() << "Shapes check failed for unexpected reason";
    }
}

TEST(type_prop, ebps_2_args_api)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::i64, Shape{3, 4});

    auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices);
    EXPECT_TRUE(ebps->get_output_partial_shape(0).same_scheme(PartialShape{3, 2}));
    EXPECT_EQ(ebps->get_output_element_type(0), element::Type_t::f32);
    EXPECT_EQ(indices->get_partial_shape().rank().get_length(), 2);
}

TEST(type_prop, ebps_fail_indices_element_type_2_args_api)
{
    auto emb_table = make_shared<op::Parameter>(element::Type_t::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});

    try
    {
        auto ebps = make_shared<op::v3::EmbeddingBagPackedSum>(emb_table, indices);
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
