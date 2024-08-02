// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/binary_convolution.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/coordinate_diff.hpp"

using namespace std;
using namespace testing;

TEST(type_prop, bin_convolution_auto_padding_same) {
    ov::PartialShape data_batch_shape{1, 1, 5, 5};
    ov::PartialShape filters_shape{1, 1, 3, 3};
    auto a_symbols = set_shape_symbols(data_batch_shape);
    auto w_symbols = set_shape_symbols(filters_shape);
    ov::Strides strides{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_batch_shape);
    auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, filters_shape);

    auto conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                           filters,
                                                           strides,
                                                           pads_begin,
                                                           pads_end,
                                                           dilations,
                                                           mode,
                                                           pad_value,
                                                           auto_pad);

    EXPECT_THAT(get_shape_symbols(conv->get_output_partial_shape(0)),
                ElementsAre(a_symbols[0], w_symbols[0], nullptr, nullptr));
    EXPECT_EQ(conv->get_output_partial_shape(0), (ov::PartialShape{1, 1, 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (ov::CoordinateDiff{1, 1}));
    EXPECT_EQ(conv->get_pads_end(), (ov::CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_lower_spatial_dims_static) {
    ov::PartialShape data_batch_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5, 5};
    ov::PartialShape filters_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 3, 3};
    auto a_symbols = set_shape_symbols(data_batch_shape);
    auto w_symbols = set_shape_symbols(filters_shape);
    ov::Strides strides{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_batch_shape);
    auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, filters_shape);

    auto conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                           filters,
                                                           strides,
                                                           pads_begin,
                                                           pads_end,
                                                           dilations,
                                                           mode,
                                                           pad_value,
                                                           auto_pad);

    EXPECT_THAT(get_shape_symbols(conv->get_output_partial_shape(0)),
                ElementsAre(a_symbols[0], w_symbols[0], nullptr, nullptr));
    EXPECT_EQ(conv->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (ov::CoordinateDiff{1, 1}));
    EXPECT_EQ(conv->get_pads_end(), (ov::CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_upper_spatial_dims_static) {
    const ov::PartialShape data_batch_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5, 5};
    const ov::PartialShape filters_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 2, 2};
    ov::Strides strides{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::SAME_UPPER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_batch_shape);
    auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, filters_shape);

    auto conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                           filters,
                                                           strides,
                                                           pads_begin,
                                                           pads_end,
                                                           dilations,
                                                           mode,
                                                           pad_value,
                                                           auto_pad);

    EXPECT_EQ(conv->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (ov::CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_pads_end(), (ov::CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_data_batch_spatial_dims_dynamic) {
    ov::PartialShape data_batch_shape{1, 1, ov::Dimension::dynamic(), 5};
    ov::PartialShape filters_shape{ov::Dimension::dynamic(), 1, 3, 3};
    auto a_symbols = set_shape_symbols(data_batch_shape);
    auto w_symbols = set_shape_symbols(filters_shape);
    ov::Strides strides{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_batch_shape);
    auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, filters_shape);

    auto conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                           filters,
                                                           strides,
                                                           pads_begin,
                                                           pads_end,
                                                           dilations,
                                                           mode,
                                                           pad_value,
                                                           auto_pad);

    EXPECT_THAT(get_shape_symbols(conv->get_output_partial_shape(0)),
                ElementsAre(a_symbols[0], w_symbols[0], nullptr, nullptr));
    EXPECT_EQ(conv->get_output_partial_shape(0),
              (ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5}));
    EXPECT_EQ(conv->get_pads_begin(), (ov::CoordinateDiff{0, 1}));
    EXPECT_EQ(conv->get_pads_end(), (ov::CoordinateDiff{0, 1}));
}

TEST(type_prop, bin_convolution_dyn_data_batch) {
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3});
    const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     ov::Strides{},
                                                                     ov::CoordinateDiff{},
                                                                     ov::CoordinateDiff{},
                                                                     ov::Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);

    EXPECT_EQ(bin_conv->get_output_partial_shape(0), (ov::PartialShape{-1, 1, {1, -1}, {1, -1}}));
}

TEST(type_prop, bin_convolution_dyn_filters) {
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
    const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape::dynamic());
    const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     ov::Strides{},
                                                                     ov::CoordinateDiff{},
                                                                     ov::CoordinateDiff{},
                                                                     ov::Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);

    EXPECT_EQ(bin_conv->get_output_partial_shape(0), (ov::PartialShape{1, -1, {1, 5}, {1, 5}}));
}

TEST(type_prop, bin_convolution_dyn_data_batch_and_filters) {
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape::dynamic());
    const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     ov::Strides{},
                                                                     ov::CoordinateDiff{},
                                                                     ov::CoordinateDiff{},
                                                                     ov::Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);

    EXPECT_EQ(bin_conv->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TEST(type_prop, bin_convolution_invalid_inputs_et) {
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;
    try {
        const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                         filters,
                                                                         ov::Strides{},
                                                                         ov::CoordinateDiff{},
                                                                         ov::CoordinateDiff{},
                                                                         ov::Strides{},
                                                                         mode,
                                                                         pad_value,
                                                                         auto_pad);
        // data batch element type must be float point
        FAIL() << "Incompatible element type of data batch input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch element type must be numeric");
    } catch (...) {
        FAIL() << "Data batch element type validation check failed for unexpected reason";
    }
    // TODO: Add test with check filters element type once u1 is supported in OpenVINO Python API
    // (#49517)
}

TEST(type_prop, bin_convolution_incompatible_input_channels) {
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
    auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 2, 3, 3});

    try {
        auto conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                               filters,
                                                               ov::Strides{},
                                                               ov::CoordinateDiff{},
                                                               ov::CoordinateDiff{},
                                                               ov::Strides{},
                                                               mode,
                                                               pad_value,
                                                               auto_pad);
        FAIL() << "Incompatible input channel dimension in data batch and filters not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch channel count"));
    } catch (...) {
        FAIL() << "Data batch and filters input channel count validation check failed for "
                  "unexpected reason";
    }
}

TEST(type_prop, bin_convolution_invalid_input_ranks) {
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 5 (Conv3D)
    try {
        const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3, 3});
        const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                         filters,
                                                                         ov::Strides{},
                                                                         ov::CoordinateDiff{},
                                                                         ov::CoordinateDiff{},
                                                                         ov::Strides{},
                                                                         mode,
                                                                         pad_value,
                                                                         auto_pad);
        // data batch and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters rank do not match");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data partial shape provided is rank 5 (Conv3D)
    // filter partial shape provided is rank 4 (Conv2D)
    try {
        const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5, 5});
        const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                         filters,
                                                                         ov::Strides{},
                                                                         ov::CoordinateDiff{},
                                                                         ov::CoordinateDiff{},
                                                                         ov::Strides{},
                                                                         mode,
                                                                         pad_value,
                                                                         auto_pad);
        // data batch and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters rank do not match");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, bin_convolution_invalid_spatial_dims_parameters) {
    ov::Strides strides_1d{1};
    ov::Strides strides_3d{1, 1, 1};

    ov::Strides dilations_2d{1, 1};
    ov::Strides dilations_3d{1, 1, 1};

    ov::CoordinateDiff pads_end_2d{0, 0};
    ov::CoordinateDiff pads_begin_3d{0, 0, 0};

    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    try {
        const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                         filters,
                                                                         strides_3d,
                                                                         ov::CoordinateDiff{},
                                                                         ov::CoordinateDiff{},
                                                                         dilations_2d,
                                                                         mode,
                                                                         pad_value,
                                                                         auto_pad);
        // ov::Strides have incompatible number of spatial dimensions
        FAIL() << "Incompatible stride number of spatial dimensions not detected.";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Strides should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Strides validation check failed for unexpected reason.";
    }

    try {
        const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                         filters,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{},
                                                                         ov::CoordinateDiff{},
                                                                         dilations_3d,
                                                                         mode,
                                                                         pad_value,
                                                                         auto_pad);
        // Dilations have incompatible number of spatial dimensions
        FAIL() << "Incompatible dilations number of spatial dimensions not detected.";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Dilations should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Dilations validation check failed for unexpected reason.";
    }

    try {
        const auto data_batch = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::u1, ov::PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<ov::op::v1::BinaryConvolution>(data_batch,
                                                                         filters,
                                                                         ov::Strides{1, 1},
                                                                         pads_begin_3d,
                                                                         pads_end_2d,
                                                                         dilations_2d,
                                                                         mode,
                                                                         pad_value,
                                                                         auto_pad);
        // Pads have incompatible number of spatial dimensions
        FAIL() << "Incompatible pads number of spatial dimensions not detected.";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Pads begin and end should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Pads validation check failed for unexpected reason.";
    }
}

class TypePropBinaryConvolutionV1Test : public TypePropOpTest<ov::op::v1::BinaryConvolution> {
protected:
    ov::CoordinateDiff empty_pad{};
};

TEST_F(TypePropBinaryConvolutionV1Test, default_ctor) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 5, 5});
    const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 4, 4});

    const auto op = make_op();
    op->set_arguments(ov::OutputVector{data, filters});
    op->set_strides({1, 3});
    op->set_dilations({1, 2});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 2});
    op->set_auto_pad(ov::op::PadType::EXPLICIT);
    op->set_mode(ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT);
    op->set_pad_value(1.0f);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_strides(), ov::Strides({1, 3}));
    EXPECT_EQ(op->get_dilations(), ov::Strides({1, 2}));
    EXPECT_EQ(op->get_pads_begin(), ov::CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_pads_end(), ov::CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({1, 2, 6, 1}));
}

TEST_F(TypePropBinaryConvolutionV1Test, interval_shapes) {
    ov::PartialShape data_batch_pshape{{1, 3}, 1, {1, 5}, {3, 10}};
    ov::PartialShape filters_pshape{2, {1, 3}, 3, 3};
    auto a_symbols = set_shape_symbols(data_batch_pshape);
    auto w_symbols = set_shape_symbols(filters_pshape);

    constexpr auto et = ov::element::f32;
    constexpr auto auto_pad = ov::op::PadType::EXPLICIT;
    constexpr auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    constexpr auto pad_value = 1.0f;

    const auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    const auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    const auto op = make_op(data_batch,
                            filters,
                            ov::Strides{},
                            ov::CoordinateDiff{},
                            ov::CoordinateDiff{},
                            ov::Strides{},
                            mode,
                            pad_value,
                            auto_pad);

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(a_symbols[0], w_symbols[0], nullptr, nullptr));
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({{1, 3}, 2, {1, 3}, {1, 8}}));
    EXPECT_EQ(op->get_pads_begin(), (ov::CoordinateDiff{0, 0}));
    EXPECT_EQ(op->get_pads_end(), (ov::CoordinateDiff{0, 0}));
}
