// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/deformable_psroi_pooling.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, deformable_psroi_pooling_default_ctor) {
    const int64_t output_dim = 48;
    const int64_t group_size = 2;

    const auto rois_dim = 30;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 4, 64, 56});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});

    auto op = make_shared<op::v1::DeformablePSROIPooling>();

    op->set_arguments(OutputVector{input_data, input_coords});
    op->set_output_dim(output_dim);
    op->set_group_size(group_size);

    op->validate_and_infer_types();

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    EXPECT_EQ(op->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_interval_labels) {
    const float spatial_scale = 0.05f;
    const int64_t output_dim = 48;
    const int64_t group_size = 2;

    const auto rois_dim = Dimension(15, 30);

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 4, 64, 56});

    auto coords_shape = PartialShape{rois_dim, 5};
    auto symbols = set_shape_symbols(coords_shape);
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, coords_shape);

    auto op =
        make_shared<op::v1::DeformablePSROIPooling>(input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    EXPECT_EQ(op->get_output_partial_shape(0), expected_output);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST(type_prop, deformable_psroi_pooling_no_offsets_group_size_3) {
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;

    const auto rois_dim = 300;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});

    auto def_psroi_pool =
        make_shared<op::v1::DeformablePSROIPooling>(input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_group_size_3) {
    const float spatial_scale = 0.0625f;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;
    const int64_t part_size = 3;
    const int64_t spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 2, part_size, part_size});

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                      input_coords,
                                                                      input_offsets,
                                                                      output_dim,
                                                                      spatial_scale,
                                                                      group_size,
                                                                      "bilinear_deformable",
                                                                      spatial_bins,
                                                                      spatial_bins,
                                                                      0.1f,
                                                                      part_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_group_size_7) {
    const float spatial_scale = 0.0625f;
    const int64_t output_dim = 162;
    const int64_t group_size = 7;
    const int64_t part_size = 7;
    const int64_t spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 2, part_size, part_size});

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                      input_coords,
                                                                      input_offsets,
                                                                      output_dim,
                                                                      spatial_scale,
                                                                      group_size,
                                                                      "bilinear_deformable",
                                                                      spatial_bins,
                                                                      spatial_bins,
                                                                      0.1f,
                                                                      part_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_dynamic_rois) {
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;

    const auto rois_dim = Dimension(100, 200);

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});

    auto def_psroi_pool =
        make_shared<op::v1::DeformablePSROIPooling>(input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_fully_dynamic) {
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;

    const auto rois_dim = Dimension::dynamic();

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto def_psroi_pool =
        make_shared<op::v1::DeformablePSROIPooling>(input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_invalid_group_size) {
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const auto rois_dim = 300;
    try {
        const int64_t group_size = 0;

        auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
        auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                          input_coords,
                                                                          output_dim,
                                                                          spatial_scale,
                                                                          group_size);

        FAIL() << "Invalid group_size not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Value of `group_size` attribute has to be greater than 0"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_output_dim) {
    const float spatial_scale = 0.0625;
    const auto rois_dim = 300;
    const int64_t group_size = 3;

    try {
        const int64_t output_dim = -882;

        auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
        auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                          input_coords,
                                                                          output_dim,
                                                                          spatial_scale,
                                                                          group_size);

        FAIL() << "Invalid output_dim not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Value of `output_dim` attribute has to be greater than 0"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_data_input_rank) {
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 162;
    const int64_t group_size = 7;
    const int64_t part_size = 7;
    const int64_t spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 2, part_size, part_size});

    try {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                          input_coords,
                                                                          input_offsets,
                                                                          output_dim,
                                                                          spatial_scale,
                                                                          group_size,
                                                                          "bilinear_deformable",
                                                                          spatial_bins,
                                                                          spatial_bins,
                                                                          0.1f,
                                                                          part_size);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid first input rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("First input rank must be compatible with 4 (input rank: 3)"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_box_coordinates_rank) {
    const int64_t output_dim = 4;
    const float spatial_scale = 0.9f;
    const int64_t group_size = 7;

    const auto rois_dim = 300;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, rois_dim, 5});
    try {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                          input_coords,
                                                                          output_dim,
                                                                          spatial_scale,
                                                                          group_size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid second input rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Second input rank must be compatible with 2 (input rank: 3)"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_offstes_rank) {
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 162;
    const int64_t group_size = 7;
    const int64_t part_size = 7;
    const int64_t spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, rois_dim, 2, part_size, part_size});
    try {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(input_data,
                                                                          input_coords,
                                                                          input_offsets,
                                                                          output_dim,
                                                                          spatial_scale,
                                                                          group_size,
                                                                          "bilinear_deformable",
                                                                          spatial_bins,
                                                                          spatial_bins,
                                                                          0.1f,
                                                                          part_size);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid third input rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Third input rank must be compatible with 4 (input rank: 5)"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
