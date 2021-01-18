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

#include "ngraph/op/proposal.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V0 ------------------------------

TEST(type_prop, proposal_v0_invalid_class_probs_rank)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto proposal =
            make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Proposal layer shape class_probs input must have rank 4"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v0_invalid_class_bbox_deltas_rank)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto proposal =
            make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Proposal layer shape class_bbox_deltas_shape input must have rank 4"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v0_invalid_image_shape_rank)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    try
    {
        auto proposal =
            make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Proposal layer image_shape input must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v0_invalid_image_shape_size)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{5});

    try
    {
        auto proposal =
            make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v0_shape_infer)
{
    op::ProposalAttrs attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{batch_size, 12, 34, 62});
    auto class_bbox_deltas =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, 24, 34, 62});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{3});
    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    ASSERT_EQ(op->get_output_shape(0), (Shape{batch_size * attrs.post_nms_topn, 5}));
}

// ------------------------------ V4 ------------------------------

TEST(type_prop, proposal_v4_invalid_class_probs_rank)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto proposal =
            make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Proposal layer shape class_probs input must have rank 4"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v4_invalid_class_bbox_deltas_rank)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto proposal =
            make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Proposal layer shape class_bbox_deltas_shape input must have rank 4"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v4_invalid_image_shape_rank)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    try
    {
        auto proposal =
            make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Proposal layer image_shape input must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v4_invalid_image_shape_size)
{
    op::ProposalAttrs attrs;
    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{5});

    try
    {
        auto proposal =
            make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, proposal_v4_shape_infer)
{
    op::ProposalAttrs attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = make_shared<op::Parameter>(element::f32, Shape{batch_size, 12, 34, 62});
    auto class_bbox_deltas =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, 24, 34, 62});
    auto image_shape = make_shared<op::Parameter>(element::f32, Shape{3});
    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    ASSERT_EQ(op->get_output_shape(0), (Shape{batch_size * attrs.post_nms_topn, 5}));
    ASSERT_EQ(op->get_output_shape(1), (Shape{batch_size * attrs.post_nms_topn}));
}
