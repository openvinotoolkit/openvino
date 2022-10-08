// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_shape_inference.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, group_convolution_auto_padding_same_lower) {
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};
    element::Type_t et = element::f32;
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    auto groupConv =
        make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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
    const PartialShape data_batch_pshape{1, Dimension::dynamic(), 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2, 2};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({1, Dimension::dynamic(), 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_static_ranks_filters_groups_dyn) {
    const PartialShape data_batch_pshape{Dimension::dynamic(), 4, 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), 1, 2, 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           auto_pad);

    ASSERT_EQ(groupConv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 2, 5, 5}));
    ASSERT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_static_ranks_filters_groups_cout_dyn) {
    const PartialShape data_batch_pshape{Dimension::dynamic(), 4, 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), 2, 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

    auto data_batch = make_shared<op::Parameter>(et, dyn_pshape);
    auto filters = make_shared<op::Parameter>(et, dyn_pshape);
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
        auto data_batch = make_shared<op::Parameter>(element::f16, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(element::f32, filters_pshape);
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
        auto data_batch = make_shared<op::Parameter>(boolean_et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(boolean_et, filters_pshape);
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
        auto filters = make_shared<op::Parameter>(et, PartialShape{2, 8, 2, 3, 3, Dimension::dynamic()});
        auto data = make_shared<op::Parameter>(et, PartialShape{1, 16, 6, 6});
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
        const auto filters = make_shared<op::Parameter>(et, PartialShape{2, 8, 2, 3, 3});
        const auto data = make_shared<op::Parameter>(et, PartialShape{1, Dimension::dynamic(), 16, 6, 6});
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
    try {
        const PartialShape data_batch_pshape{1, 6, 5, 5};
        const PartialShape filters_pshape{2, 1, 2, 3, 3};
        element::Type_t et = element::f32;

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                               filters,
                                                               Strides{},
                                                               CoordinateDiff{},
                                                               CoordinateDiff{},
                                                               Strides{});
        // data batch shape does not have correct dimension C_IN * GROUPS
        FAIL() << "Invalid input channels dimension of data batch not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input channels dimension of data batch has incompatible value "
                             "with filter shape.");
    } catch (...) {
        FAIL() << "Input channels dimension of data batch validation check failed for unexpected "
                  "reason.";
    }

    try {
        const PartialShape data_batch_pshape{1, 3, 5, 5};
        const PartialShape filters_pshape{2, 1, Dimension::dynamic(), 3, 3};
        element::Type_t et = element::f32;

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(data_batch,
                                                               filters,
                                                               Strides{},
                                                               CoordinateDiff{},
                                                               CoordinateDiff{},
                                                               Strides{});
        // data batch shape does not have correct dimension C_IN * GROUPS
        FAIL() << "Invalid input channels dimension of data batch not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Input channels dimension of data batch not a multiple of group size");
    } catch (...) {
        FAIL() << "Input channels dimension of data batch validation check failed for unexpected "
                  "reason.";
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

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
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

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
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

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
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

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dilations should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }

    // invalid padding spatial dimensions
    try {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Pads begin should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
    try {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0};

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv =
            make_shared<op::v1::GroupConvolution>(data_batch, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Pads end should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
}

TEST(type_prop, group_convolution_default_constructed) {
    auto conv = make_shared<op::v1::GroupConvolution>();
    conv->set_auto_pad(op::PadType::SAME_LOWER);

    const auto &input_shape = ov::PartialShape::dynamic(), filters_shape = ov::PartialShape{1, 1, 1, 3, 3};
    const auto& input_shapes = std::vector<ov::PartialShape>{input_shape, filters_shape};
    std::vector<ov::PartialShape> output_shapes(1);
    auto pad_begin = CoordinateDiff{}, pad_end = CoordinateDiff{};

    int64_t num_spatial = calculate_num_spatial(conv.get(), input_shape, filters_shape, 2, 3);
    update_and_validate_attributes(conv.get(), num_spatial);
    EXPECT_NO_THROW(shape_infer(conv.get(), pad_begin, pad_end, input_shapes, output_shapes));
}