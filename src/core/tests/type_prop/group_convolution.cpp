// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/group_conv.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, group_convolution_auto_padding_same_lower) {
    PartialShape data_batch_pshape{1, 4, 5, 5};
    PartialShape filters_pshape{2, 1, 2, 3, 3};
    auto data_symbols = set_shape_symbols(data_batch_pshape);
    auto filter_symbols = set_shape_symbols(filters_pshape);
    element::Type_t et = element::f32;
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);

    auto groupConv =
        make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_THAT(get_shape_symbols(groupConv->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], filter_symbols[0], nullptr, nullptr));
    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({1, 2, 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_upper) {
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 2, 2};
    element::Type_t et = element::f32;
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);

    auto conv =
        make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0), PartialShape({1, 2, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_lower_spatial_dims_static) {
    const PartialShape data_batch_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_upper_spatial_dims_static) {
    PartialShape data_batch_pshape{1, Dimension::dynamic(), 5, 5};
    PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2, 2};
    auto data_symbols = set_shape_symbols(data_batch_pshape);
    auto filter_symbols = set_shape_symbols(filters_pshape);
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    EXPECT_THAT(get_shape_symbols(groupConv->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], nullptr, nullptr, nullptr));
    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({1, Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_static_ranks_filters_groups_dyn) {
    PartialShape data_batch_pshape{Dimension::dynamic(), 4, 5, 5};
    PartialShape filters_pshape{Dimension::dynamic(), 1, 2, 3, 3};
    auto data_symbols = set_shape_symbols(data_batch_pshape);
    auto filter_symbols = set_shape_symbols(filters_pshape);

    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);
    EXPECT_THAT(get_shape_symbols(groupConv->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], filter_symbols[0], nullptr, nullptr));
    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 2, 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_static_ranks_filters_groups_cout_dyn) {
    const PartialShape data_batch_pshape{Dimension::dynamic(), 4, 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), 2, 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_static_ranks_data_cin_filters_group_dyn) {
    const PartialShape data_batch_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), 1, 2, 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_spatial_dims_dynamic) {
    const PartialShape data_batch_pshape{1, 4, Dimension::dynamic(), 5};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({1, 2, Dimension::dynamic(), 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 1}));
}

TEST(type_prop, group_convolution_data_batch_dynamic) {
    const PartialShape data_batch_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{});

    ASSERT_EQ(groupConv->get_auto_pad(), op::PadType::EXPLICIT);
    ASSERT_EQ(groupConv->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(groupConv->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_output_partial_shape(0),
              PartialShape({Dimension::dynamic(), 2, Dimension(1, -1), Dimension(1, -1)}));
}

TEST(type_prop, group_convolution_filters_dynamic_auto_pad_explicit) {
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f16;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{});

    ASSERT_EQ(groupConv->get_auto_pad(), op::PadType::EXPLICIT);
    ASSERT_EQ(groupConv->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(groupConv->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_output_partial_shape(0),
              PartialShape({1, Dimension::dynamic(), Dimension{1, 5}, Dimension{1, 5}}));
}

TEST(type_prop, group_convolution_filters_dynamic_auto_pad_same) {
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f16;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_auto_pad(), op::PadType::SAME_LOWER);
    // pads should be as default since filters shape is dynamic
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({1, Dimension::dynamic(), 5, 5}));
}

TEST(type_prop, group_convolution_data_batch_and_filters_dynamic) {
    const PartialShape dyn_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, dyn_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, dyn_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{});

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, group_convolution_invalid_et_inputs) {
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};

    try {
        auto data_batch = make_shared<ov::op::v0::Parameter>(element::f16, data_batch_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(element::f32, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                               filters,
                                                               Strides{},
                                                               CoordinateDiff{},
                                                               CoordinateDiff{},
                                                               Strides{});
        // data batch and filters must be of same element type
        FAIL() << "Invalid element type of inputs not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element types for data batch and filters do not match");
    } catch (...) {
        FAIL() << "Element types of data batch and filters validation check failed for unexpected "
                  "reason.";
    }

    try {
        const element::Type boolean_et = element::boolean;
        auto data_batch = make_shared<ov::op::v0::Parameter>(boolean_et, data_batch_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(boolean_et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                               filters,
                                                               Strides{},
                                                               CoordinateDiff{},
                                                               CoordinateDiff{},
                                                               Strides{});
        // data batch and filters must be of numeric element type
        FAIL() << "Boolean element type of inputs not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of inputs must be numeric");
    } catch (...) {
        FAIL() << "Numeric element types of data batch and filters validation check failed for "
                  "unexpected reason.";
    }
}

TEST(type_prop, group_convolution_invalid_input_ranks) {
    const element::Type_t et = element::f32;

    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 6 (Conv3D)
    try {
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape{2, 8, 2, 3, 3, Dimension::dynamic()});
        auto data = make_shared<ov::op::v0::Parameter>(et, PartialShape{1, 16, 6, 6});
        auto groupConv = make_shared<op::v1::GroupConvolution>(data,
                                                               filters,
                                                               Strides{},
                                                               CoordinateDiff{},
                                                               CoordinateDiff{},
                                                               Strides{});
        // data and weight have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch and filters rank do not match"));
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data partial shape provided is rank 5 (Conv3D)
    // filter partial shape provided is rank 5 (Conv2D)
    try {
        const auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape{2, 8, 2, 3, 3});
        const auto data = make_shared<ov::op::v0::Parameter>(et, PartialShape{1, Dimension::dynamic(), 16, 6, 6});
        const auto groupConv = make_shared<op::v1::GroupConvolution>(data,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{});
        // data and weight have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch and filters rank do not match"));
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, group_convolution_invalid_input_channel_dims) {
    constexpr auto et = element::f32;
    // data batch shape does not have correct dimension C_IN * GROUPS
    {
        const PartialShape data_batch_pshape{1, 6, 5, 5};
        const PartialShape filters_pshape{1, 1, 3, 3, 3};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);

        OV_EXPECT_THROW(
            const auto op = make_shared<op::v1::GroupConvolution>(data_batch,
                                                                  filters,
                                                                  Strides{},
                                                                  CoordinateDiff{},
                                                                  CoordinateDiff{},
                                                                  Strides{}),
            NodeValidationFailure,
            HasSubstr("Input channels dimension of data batch is incompatible with filter groups or input channels."));
    }

    // data batch shape does not have correct dimension C_IN * GROUPS
    {
        const PartialShape data_batch_pshape{1, 3, 5, 5};
        const PartialShape filters_pshape{-1, 1, 2, 3, 3};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);

        OV_EXPECT_THROW(
            const auto op = make_shared<op::v1::GroupConvolution>(data_batch,
                                                                  filters,
                                                                  Strides{},
                                                                  CoordinateDiff{},
                                                                  CoordinateDiff{},
                                                                  Strides{}),
            NodeValidationFailure,
            HasSubstr("Input channels dimension of data batch is incompatible with filter groups or input channels."));
    }
}

TEST(type_prop, group_convolution_invalid_conv_param_spatial_dims) {
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 2, 2};
    const element::Type_t et = element::f32;

    // invalid strides spatial dimensions
    try {
        Strides strides{1, 1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Strides should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }
    try {
        Strides strides{1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Strides should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }

    // invalid dilations spatial dimensions
    try {
        Strides strides{1, 1};
        Strides dilations{1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dilations should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }
    try {
        Strides strides{1, 1};
        Strides dilations{1, 1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dilations should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }

    // invalid padding spatial dimensions
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);

        OV_EXPECT_THROW(
            auto op =
                make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations),
            NodeValidationFailure,
            HasSubstr("Pads begin and end should be defined for all and only spatial dimensions."));
    }

    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0};

        auto data_batch = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);

        OV_EXPECT_THROW(
            auto op =
                make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations),
            NodeValidationFailure,
            HasSubstr("Pads begin and end should be defined for all and only spatial dimensions."));
    }
}

TEST(type_prop, group_convolution_interval_shapes) {
    PartialShape data_batch_pshape{{1, 3}, {2, 6}, {1, 5}, {3, 10}, {20, 100}};
    PartialShape filters_pshape{{2, 3}, {1, 3}, {2, 3}, 3, 3, 3};
    auto data_symbols = set_shape_symbols(data_batch_pshape);
    set_shape_symbols(filters_pshape);

    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::EXPLICIT;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);
    EXPECT_THAT(get_shape_symbols(groupConv->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], nullptr, nullptr, nullptr, nullptr));
    EXPECT_EQ(groupConv->get_output_partial_shape(0), PartialShape({{1, 3}, {2, 9}, {1, 3}, {1, 8}, {18, 98}}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, group_convolution_default_constructed) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 1, 1, 3, 3});

    const auto op = make_shared<op::v1::GroupConvolution>();
    op->set_arguments(OutputVector{data, filters});
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 2});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_strides(), Strides({1, 1}));
    EXPECT_EQ(op->get_dilations(), Strides({1, 1}));
    EXPECT_EQ(op->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_pads_end(), CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 1, {2, -1}, {2, -1}}));
}
