// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, gather_tree_output_shape)
{
    auto step_ids = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto parent_idx = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto max_seq_len = make_shared<op::Parameter>(element::i64, Shape{1});
    auto end_token = make_shared<op::Parameter>(element::i64, Shape{});

    auto gather_tree =
        make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);

    ASSERT_EQ(gather_tree->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(gather_tree->get_output_element_type(0), element::i64);
}

TEST(type_prop, gather_tree_pooling_step_ids_invalid_rank)
{
    auto step_ids = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3, 4});
    auto parent_idx = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto max_seq_len = make_shared<op::Parameter>(element::i64, Shape{1});
    auto end_token = make_shared<op::Parameter>(element::i64, Shape{});
    try
    {
        auto gather_tree =
            make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid step_ids input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("step_ids input rank must equal to 3 (step_ids rank: 4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_tree_parent_idx_invalid_rank)
{
    auto step_ids = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto parent_idx = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3, 4});
    auto max_seq_len = make_shared<op::Parameter>(element::i64, Shape{1});
    auto end_token = make_shared<op::Parameter>(element::i64, Shape{});
    try
    {
        auto gather_tree =
            make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid parent_idx input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("parent_idx input rank must equal to 3 (parent_idx rank: 4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_tree_max_seq_len_invalid_rank)
{
    auto step_ids = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto parent_idx = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto max_seq_len = make_shared<op::Parameter>(element::i64, Shape{1, 2});
    auto end_token = make_shared<op::Parameter>(element::i64, Shape{});
    try
    {
        auto gather_tree =
            make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid parent_idx input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("max_seq_len input rank must equal to 1 (max_seq_len rank: 2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_tree_end_token_invalid_rank)
{
    auto step_ids = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto parent_idx = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto max_seq_len = make_shared<op::Parameter>(element::i64, Shape{1});
    auto end_token = make_shared<op::Parameter>(element::i64, Shape{1});
    try
    {
        auto gather_tree =
            make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid end_token input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("end_token input rank must be scalar (end_token rank: 1)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
