// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "util/type_prop.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/prior_box_clustered.hpp"

using namespace ngraph;


TEST(type_prop, prior_box_clustered)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f, 1.1f};

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {19, 19});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
    auto pbc = std::make_shared<op::PriorBoxClustered>(layer_shape, image_shape, attrs);
    // Output shape - 4 * 19 * 19 * 3 (attrs.widths.size())
    ASSERT_EQ(pbc->get_shape(), (Shape{2, 4332}));
}

TEST(type_prop, prior_box_clustered_float_layer_shape)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f, 1.1f};

    auto layer_shape = op::Constant::create<float>(element::f32, Shape{2}, {19, 19});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});

    try
    {
        auto pbc = std::make_shared<op::PriorBoxClustered>(layer_shape, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect prior_box_clustered value type exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("layer shape input must be an integral number"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, prior_box_clustered_float_image_shape)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f, 1.1f};

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {19, 19});
    auto image_shape = op::Constant::create<float>(element::f32, Shape{2}, {300, 300});

    try
    {
        auto pbc = std::make_shared<op::PriorBoxClustered>(layer_shape, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect prior_box_clustered value type exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("image shape input must be an integral number"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, prior_box_clustered_widths_heights_different)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f};

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {19, 19});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});

    try
    {
        auto pbc = std::make_shared<op::PriorBoxClustered>(layer_shape, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect prior_box_clustered value type exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Size of heights vector:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, prior_box_clustered_not_rank_2)
{
    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f, 1.1f};

    auto layer_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {19, 19, 19});
    auto image_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});

    try
    {
        auto pbc = std::make_shared<op::PriorBoxClustered>(layer_shape, image_shape, attrs);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect prior_box_clustered value type exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Layer shape must have rank 2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
