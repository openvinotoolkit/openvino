// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, deformable_convolution_partial_auto_padding_same)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 36, 5, 5};
    const PartialShape filters_pshape{4, 1, 3, 3};
    const element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 4, 5, 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, deformable_convolution_partial_auto_padding_same_lower_data_batch_nc_dims_dynamic)
{
    const PartialShape data_batch_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape offsets_pshape{Dimension::dynamic(), 36, 5, 5};
    const PartialShape filters_pshape{4, 4, 3, 3};
    const element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 4, 5, 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, deformable_convolution_partial_auto_padding_same_upper_data_batch_nc_dims_dynamic)
{
    const PartialShape data_batch_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape offsets_pshape{1, 16, 5, 5};
    const PartialShape filters_pshape{4, 4, 2, 2};
    const element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 4, 5, 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, deformable_convolution_partial_auto_padding_same_spatial_dims_dynamic)
{
    const PartialShape data_batch_pshape{1, 4, Dimension::dynamic(), 5};
    const PartialShape offsets_pshape{1, 36, 5, 5};
    const PartialShape filters_pshape{4, 1, 3, 3};
    const element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(
        deformable_conv->get_output_partial_shape(0).same_scheme({1, 4, Dimension::dynamic(), 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{0, 1}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{0, 1}));
}

TEST(type_prop, deformable_convolution_data_batch_dynamic)
{
    const PartialShape data_batch_pshape{PartialShape::dynamic()};
    const PartialShape offsets_pshape{2, 36, 5, 5};
    const PartialShape filters_pshape{4, 4, 3, 3};
    const element::Type_t et = element::f32;

    const auto auto_pad = op::PadType::EXPLICIT;
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_EQ(deformable_conv->get_auto_pad(), op::PadType::EXPLICIT);
    ASSERT_EQ(deformable_conv->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{0, 0}));
    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{2, 4, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, deformable_convolution_offsets_dynamic)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{4, 2, 3, 3};
    const element::Type_t et = element::f32;

    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_EQ(deformable_conv->get_auto_pad(), op::PadType::SAME_LOWER);
    ASSERT_EQ(deformable_conv->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 4, 5, 5}));
}

TEST(type_prop, deformable_convolution_auto_pad_same_filters_dynamic)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 36, 3, 3};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    const auto auto_pad = op::PadType::SAME_UPPER;
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_EQ(deformable_conv->get_auto_pad(), op::PadType::SAME_UPPER);
    ASSERT_EQ(deformable_conv->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{0, 0}));
    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, deformable_convolution_deformable_data_batch_and_filters_dynamic)
{
    const PartialShape data_batch_pshape{PartialShape::dynamic()};
    const PartialShape offsets_pshape{1, 36, 3, 3};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    const auto auto_pad = op::PadType::EXPLICIT;
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_EQ(deformable_conv->get_auto_pad(), op::PadType::EXPLICIT);
    ASSERT_EQ(deformable_conv->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{0, 0}));
    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, deformable_convolution_deformable_all_inputs_dynamic)
{
    const PartialShape dyn_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    const auto auto_pad = op::PadType::EXPLICIT;
    const int64_t group = 2;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, dyn_pshape);
    auto offsets = make_shared<op::Parameter>(et, dyn_pshape);
    auto filters = make_shared<op::Parameter>(et, dyn_pshape);
    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_EQ(deformable_conv->get_auto_pad(), op::PadType::EXPLICIT);
    ASSERT_EQ(deformable_conv->get_strides(), (Strides{}));
    ASSERT_EQ(deformable_conv->get_dilations(), (Strides{}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{}));
    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, deformable_convolution_invalid_et_inputs)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 4, 3, 3};
    const PartialShape filters_pshape{4, 4, 3, 3};

    element::Type_t float_et = element::f32;
    element::Type_t boolean_et = element::boolean;

    try
    {
        auto data_batch = make_shared<op::Parameter>(element::f16, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(float_et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(float_et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // data batch input must be of same element type as filters and deformable values
        FAIL() << "Invalid element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element types of inputs do not match. Got: data batch (f16), "
                             "offsets (f32) and filters (f32)");
    }
    catch (...)
    {
        FAIL() << "Element types of inputs validation check failed for unexpected reason.";
    }

    try
    {
        auto data_batch = make_shared<op::Parameter>(float_et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(float_et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(element::f16, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // filters input must be of same element type as data batch and deformable values
        FAIL() << "Invalid element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element types of inputs do not match. Got: "
                             "data batch (f32), offsets (f32) and filters (f16)");
    }
    catch (...)
    {
        FAIL() << "Element types of inputs validation check failed for unexpected reason.";
    }

    try
    {
        auto data_batch = make_shared<op::Parameter>(float_et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(element::f16, offsets_pshape);
        auto filters = make_shared<op::Parameter>(float_et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // deformable values input must be of same element type as data batch and filters
        FAIL() << "Invalid element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element types of inputs do not match. Got: data batch (f32), "
                             "offsets (f16) and filters (f32)");
    }
    catch (...)
    {
        FAIL() << "Element types of inputs validation check failed for unexpected reason.";
    }

    try
    {
        auto data_batch = make_shared<op::Parameter>(boolean_et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(boolean_et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(boolean_et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // element types for inputs must be numeric
        FAIL() << "Invalid boolean element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of inputs must be numeric");
    }
    catch (...)
    {
        FAIL() << "Numeric element types of inputs validation check failed for "
                  "unexpected reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_input_ranks)
{
    const element::Type_t et = element::f32;

    // data batch shape provides is rank 5
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5, 5, 5};
        const PartialShape offsets_pshape{1, 4, 4, 4};
        const PartialShape filters_pshape{4, 4, 3, 3};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // data batch has invalid rank 5, should be 4
        FAIL() << "Incompatible data batch input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Ranks of inputs do not match. Got: data batch "
                             "shape {1,4,5,5,5}, offsets shape {1,4,4,4}, filters "
                             "shape {4,4,3,3}");
    }
    catch (...)
    {
        FAIL() << "Rank validation check of data batch input failed for unexpected reason";
    }

    // deformable values shape provides is rank 5
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5, 5};
        const PartialShape offsets_pshape{1, 4, 4, 4, 4};
        const PartialShape filters_pshape{4, 4, 3, 3};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // deformable values has invalid rank 5, should be 4
        FAIL() << "Incompatible offsets input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Ranks of inputs do not match. Got: data batch shape "
                             "{1,4,5,5}, offsets shape {1,4,4,4,4}, filters shape "
                             "{4,4,3,3}");
    }
    catch (...)
    {
        FAIL() << "Rank validation check of offsets input failed for unexpected reason";
    }

    // filter partial shape provided is rank 5
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5, 5};
        const PartialShape offsets_pshape{1, 4, 4, 4};
        const PartialShape filters_pshape{4, 4, 3, 3, 3};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // filters has invalid rank 5, should be 4
        FAIL() << "Incompatible filter input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Ranks of inputs do not match. Got: data batch shape "
                             "{1,4,5,5}, offsets shape {1,4,4,4}, filters shape "
                             "{4,4,3,3,3}");
    }
    catch (...)
    {
        FAIL() << "Rank validation check of filter input failed for unexpected reason";
    }

    // 3D deformable convolution (not supported)
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5, 5, 5};
        const PartialShape offsets_pshape{1, 4, 4, 4, 4};
        const PartialShape filters_pshape{4, 4, 3, 3, 3};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // inputs have rank 5, should be 4
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Inputs must be of rank 4");
    }
    catch (...)
    {
        FAIL() << "Rank validation check for 2 spatial dimension inputs failed for unexpected reason";
    }

    // 1D deformable convolution (not supported)
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5};
        const PartialShape offsets_pshape{1, 4, 4};
        const PartialShape filters_pshape{4, 4, 3};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{});
        // inputs have rank 3, should be 4
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Inputs must be of rank 4");
    }
    catch (...)
    {
        FAIL() << "Rank validation check for 2 spatial dimension inputs failed for unexpected reason";
    }
}

TEST(type_prop, deformable_convolution_invalid_groups)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 4, 3, 3};
    const PartialShape filters_pshape{4, 4, 3, 3};
    element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 0;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // attribute group is invalid
        FAIL() << "Invalid attribute group value not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Attribute 'group' must be any value starting from 1");
    }
    catch (...)
    {
        FAIL() << "Attribute group validation check failed for unexpected "
                  "reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_deformable_groups)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 4, 3, 3};
    const PartialShape filters_pshape{4, 4, 3, 3};
    element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 0;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // attribute deformable group is invalid
        FAIL() << "Invalid attribute deformable group value not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Attribute 'deformable group' must be any value starting from 1");
    }
    catch (...)
    {
        FAIL() << "Attribute deformable group validation check failed for unexpected "
                  "reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_offsets_channels_dim)
{
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5, 5};
        const PartialShape offsets_pshape{1, 9, 3, 3};
        const PartialShape filters_pshape{4, 4, 3, 3};
        element::Type_t et = element::f32;

        Strides strides{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};
        Strides dilations{1, 1};
        const auto auto_pad = op::PadType::SAME_LOWER;
        const int64_t group = 4;
        const int64_t deformable_group = 2;

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // Channels dim of deformable values is incorrect. Should be 36
        FAIL() << "Invalid channels dimension of offsets input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The channels dimension of offsets input is not "
                             "compatible with filters and 'deformable group' attribute");
    }
    catch (...)
    {
        FAIL() << "Channels dimension of offsets input validation check failed for "
                  "unexpected "
                  "reason.";
    }

    // filters spatial dims are dynamic
    // we can still check if channels dim of offsets if evenly
    // divisible by deformable group attribute
    try
    {
        const PartialShape data_batch_pshape{1, 4, 5, 5};
        const PartialShape offsets_pshape{1, 35, Dimension::dynamic(), Dimension::dynamic()};
        const PartialShape filters_pshape{4, 4, Dimension::dynamic(), Dimension::dynamic()};
        element::Type_t et = element::f32;

        Strides strides{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};
        Strides dilations{1, 1};
        const auto auto_pad = op::PadType::SAME_LOWER;
        const int64_t group = 4;
        const int64_t deformable_group = 2;

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // Channels dim of deformable values is incorrect
        FAIL() << "Invalid channels dimension of offsets input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The channels dimension of offsets input must be "
                             "evenly divisible by the 'deformable group' value along the "
                             "channels axis.");
    }
    catch (...)
    {
        FAIL() << "Channels dimension of offsets input validation check failed for "
                  "unexpected reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_offsets_batch_dim)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{2, 36, 3, 3};
    const PartialShape filters_pshape{4, 4, 3, 3};
    element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // data batch and deformable values inputs must have the same batch dimension
        FAIL() << "Invalid batch dimension of offsets input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch and offsets batch dimension must be same value");
    }
    catch (...)
    {
        FAIL()
            << "Batch dimension of offsets input validation check failed for unexpected "
               "reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_data_batch_channels_dim_with_group)
{
    const PartialShape data_batch_pshape{1, 5, 5, 5};
    const PartialShape offsets_pshape{1, 36, 3, 3};
    const PartialShape filters_pshape{4, 5, 3, 3};
    element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // data batch channels is not evenly divisible by the attribute group value
        FAIL() << "Invalid channels dimension of data batch input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The input data shape must be evenly divisible by the 'group' value "
                             "along the channels axis.");
    }
    catch (...)
    {
        FAIL() << "Data batch channel dimension validation check failed for unexpected "
                  "reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_filters_channels_dim_with_group)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 36, 3, 3};
    const PartialShape filters_pshape{5, 4, 3, 3};
    element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // filters channels output is not evenly divisible by the attribute group value
        FAIL() << "Invalid channels output dimension of filters input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "The filters shape must be evenly divisible by the 'group' value along "
            "the channels axis");
    }
    catch (...)
    {
        FAIL() << "Filters channels output dimension validation check failed for unexpected "
                  "reason.";
    }
}

TEST(type_prop, deformable_convolution_incompatible_data_batch_and_filters_channels_dim)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 36, 3, 3};
    const PartialShape filters_pshape{4, 4, 3, 3};
    element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                          offsets,
                                                                          filters,
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          group,
                                                                          deformable_group);
        // data batch and filters should have same channels input dimension
        FAIL() << "Incompatible channels dimension of data batch and filters input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Data batch channel count (4) does not match filter input channel count (16)");
    }
    catch (...)
    {
        FAIL() << "Data batch channel and filter channel dimension validation check failed for "
                  "unexpected "
                  "reason.";
    }
}

TEST(type_prop, deformable_convolution_invalid_offsets_spatial_dims)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 36, 4, 4};
    const PartialShape filters_pshape{4, 1, 3, 3};
    const element::Type_t et = element::f32;

    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    try
    {
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      offsets,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);
        // deformable values has incorrect spatial dimensions                                                                      
        FAIL() << "Invalid spatial dimensions of offsets not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Spatial dimensions of offsets and output must be equal");
    }
    catch (...)
    {
        FAIL() << "Spatial dimension of offsets validation check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_convolution_invalid_conv_param_spatial_dims)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape offsets_pshape{1, 18, 3, 3};
    const PartialShape filters_pshape{4, 4, 3, 3};
    element::Type_t et = element::f32;

    // invalid strides spatial dimensions
    try
    {
        Strides strides{1, 1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(
            data_batch, offsets, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Strides should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(
            data_batch, offsets, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Strides should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }

    // invalid dilations spatial dimensions
    try
    {
        Strides strides{1, 1};
        Strides dilations{1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(
            data_batch, offsets, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Dilations should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(
            data_batch, offsets, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Dilations should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }

    // invalid padding spatial dimensions
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(
            data_batch, offsets, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Pads should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0};

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto offsets = make_shared<op::Parameter>(et, offsets_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto deformable_conv = make_shared<op::v1::DeformableConvolution>(
            data_batch, offsets, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Pads should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
}
