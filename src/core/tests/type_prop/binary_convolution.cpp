// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, bin_convolution_auto_padding_same) {
    const PartialShape data_batch_shape{1, 1, 5, 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 1, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_lower_spatial_dims_static) {
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_upper_spatial_dims_static) {
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_data_batch_spatial_dims_dynamic) {
    const PartialShape data_batch_shape{1, 1, Dimension::dynamic(), 5};
    const PartialShape filters_shape{Dimension::dynamic(), 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({1, Dimension::dynamic(), Dimension::dynamic(), 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 1}));
}

TEST(type_prop, bin_convolution_dyn_data_batch) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3});
    const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                 filters,
                                                                 Strides{},
                                                                 CoordinateDiff{},
                                                                 CoordinateDiff{},
                                                                 Strides{},
                                                                 mode,
                                                                 pad_value,
                                                                 auto_pad);
    ASSERT_TRUE(bin_conv->get_output_partial_shape(0).rank().same_scheme(Rank{3}));
    ASSERT_TRUE(
        bin_conv->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()}));
}

TEST(type_prop, bin_convolution_dyn_filters) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::u1, PartialShape::dynamic());
    const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                 filters,
                                                                 Strides{},
                                                                 CoordinateDiff{},
                                                                 CoordinateDiff{},
                                                                 Strides{},
                                                                 mode,
                                                                 pad_value,
                                                                 auto_pad);
    ASSERT_TRUE(bin_conv->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(bin_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, bin_convolution_dyn_data_batch_and_filters) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::u1, PartialShape::dynamic());
    const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                 filters,
                                                                 Strides{},
                                                                 CoordinateDiff{},
                                                                 CoordinateDiff{},
                                                                 Strides{},
                                                                 mode,
                                                                 pad_value,
                                                                 auto_pad);
    ASSERT_TRUE(bin_conv->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bin_conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, bin_convolution_invalid_inputs_et) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;
    try {
        const auto data_batch = make_shared<op::Parameter>(element::boolean, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // data batch element type must be float point
        FAIL() << "Incompatible element type of data batch input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch element type must be numeric");
    } catch (...) {
        FAIL() << "Data batch element type validation check failed for unexpected reason";
    }
    // TODO: Add test with check filters element type once u1 is supported in nGraph Python API
    // (#49517)
}

TEST(type_prop, bin_convolution_incompatible_input_channels) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
    auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 2, 3, 3});

    try {
        auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           mode,
                                                           pad_value,
                                                           auto_pad);
        FAIL() << "Incompatible input channel dimension in data batch and filters not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch channel count"));
    } catch (...) {
        FAIL() << "Data batch and filters input channel count validation check failed for "
                  "unexpected reason";
    }
}

TEST(type_prop, bin_convolution_invalid_input_ranks) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 5 (Conv3D)
    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // data batch and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters inputs must have same rank");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data partial shape provided is rank 5 (Conv3D)
    // filter partial shape provided is rank 4 (Conv2D)
    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // data batch and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters inputs must have same rank");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, bin_convolution_invalid_spatial_dims_parameters) {
    Strides strides_1d{1};
    Strides strides_3d{1, 1, 1};

    Strides dilations_2d{1, 1};
    Strides dilations_3d{1, 1, 1};

    CoordinateDiff pads_end_2d{0, 0};
    CoordinateDiff pads_begin_3d{0, 0, 0};

    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     strides_3d,
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     dilations_2d,
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // Strides have incompatible number of spatial dimensions
        FAIL() << "Incompatible stride number of spatial dimensions not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Strides should be defined for all and only spatial features."));
    } catch (...) {
        FAIL() << "Strides validation check failed for unexpected reason.";
    }

    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     strides_1d,
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     dilations_2d,
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // Dilations have incompatible number of spatial dimensions
        FAIL() << "Incompatible dilations number of spatial dimensions not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Dilations should be defined for all and only spatial features."));
    } catch (...) {
        FAIL() << "Dilations validation check failed for unexpected reason.";
    }

    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     strides_3d,
                                                                     pads_begin_3d,
                                                                     pads_end_2d,
                                                                     dilations_3d,
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // Pads have incompatible number of spatial dimensions
        FAIL() << "Incompatible pads number of spatial dimensions not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Pads should be defined for all and only spatial features."));
    } catch (...) {
        FAIL() << "Pads validation check failed for unexpected reason.";
    }
}
