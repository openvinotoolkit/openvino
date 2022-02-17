// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/psroi_pooling.hpp"

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

TEST(type_prop, psroi_pooling_average) {
    auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 4, 5});
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
    auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
    ASSERT_EQ(op->get_shape(), (Shape{150, 2, 6, 6}));
    ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
}

TEST(type_prop, psroi_pooling_bilinear) {
    auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 4, 5});
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
    auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 18, 6, 1, 2, 2, "bilinear");
    ASSERT_EQ(op->get_shape(), (Shape{150, 18, 6, 6}));
    ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
}

TEST(type_prop, psroi_pooling_invalid_type) {
    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::i32, Shape{1, 72, 4, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Feature maps' data type must be floating point"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 4, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::i32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Coords' data type must be floating point"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, psroi_pooling_invalid_mode) {
    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 4, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "invalid_mode");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Expected 'average' or 'bilinear' mode"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, psroi_pooling_invalid_shapes) {
    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("PSROIPooling expects 4 dimensions for input"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 1, 72, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("PSROIPooling expects 2 dimensions for box coordinates"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, psroi_pooling_invalid_group_size) {
    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 0, 1, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("group_size has to be greater than 0"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 5, 1, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of input's channels must be a multiply of group_size * group_size"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, psroi_pooling_invalid_output_dim) {
    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 17, 2, 1, 0, 0, "average");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("output_dim must be equal to input channels divided by group_size * group_size"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, psroi_pooling_invalid_spatial_bins) {
    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 17, 2, 1, 0, 0, "bilinear");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("spatial_bins_x has to be greater than 0"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 17, 2, 1, 1, 0, "bilinear");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("spatial_bins_y has to be greater than 0"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 17, 2, 1, 2, 5, "bilinear");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of input's channels must be a multiply of "
                                         "spatial_bins_x * spatial_bins_y"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    try {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 5, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 10, 2, 1, 2, 4, "bilinear");
        FAIL() << "Exception expected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("output_dim must be equal to input channels divided by "
                                         "spatial_bins_x * spatial_bins_y"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, psroi_pooling_dynamic_ranks) {
    {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{150, 5});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        ASSERT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
        ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
    }
    {
        auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 4, 5});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        ASSERT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
        ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
    }
}

TEST(type_prop, psroi_pooling_dynamic_num_boxes) {
    auto inputs = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 72, 4, 5});
    auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, PartialShape{{Dimension::dynamic(), 5}});
    auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{{Dimension::dynamic(), 2, 6, 6}}));
    ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
}

TEST(type_prop, psroi_pooling_static_rank_dynamic_shape) {
    {
        auto inputs = std::make_shared<op::Parameter>(
            element::Type_t::f32,
            PartialShape{{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32,
                                                      PartialShape{{Dimension::dynamic(), Dimension::dynamic()}});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{{Dimension::dynamic(), 2, 6, 6}}));
        ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
    }
    {
        auto inputs = std::make_shared<op::Parameter>(
            element::Type_t::f32,
            PartialShape{{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}});
        auto coords = std::make_shared<op::Parameter>(element::Type_t::f32, PartialShape{{200, Dimension::dynamic()}});
        auto op = std::make_shared<op::PSROIPooling>(inputs, coords, 2, 6, 0.0625, 0, 0, "average");
        ASSERT_EQ(op->get_shape(), (Shape{200, 2, 6, 6}));
        ASSERT_EQ(op->get_element_type(), element::Type_t::f32);
    }
}
