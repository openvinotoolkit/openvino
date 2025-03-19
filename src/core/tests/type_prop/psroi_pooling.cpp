// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropPSROIPoolingV0 : public TypePropOpTest<op::v0::PSROIPooling> {
protected:
    float spatial_scale = 0.625f;
    int bin_not_used = 0;
    Shape pooling_roi_2x2{2, 2};
};

TEST_F(TypePropPSROIPoolingV0, basic_average) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, bin_not_used, bin_not_used, "average");

    EXPECT_EQ(op->get_shape(), (Shape{150, 2, 6, 6}));
    EXPECT_EQ(op->get_element_type(), element::f32);
}

TEST_F(TypePropPSROIPoolingV0, basic_bilinear) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    auto op = make_op(inputs, coords, 18, 6, 1.0f, 2, 2, "bilinear");

    EXPECT_EQ(op->get_shape(), (Shape{150, 18, 6, 6}));
    EXPECT_EQ(op->get_element_type(), element::f32);
}

TEST_F(TypePropPSROIPoolingV0, invalid_features_element_type) {
    const auto inputs = std::make_shared<Parameter>(element::i32, Shape{1, 72, 4, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 6, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("Feature maps' data type must be floating point"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_rois_element_type) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 5});
    const auto coords = std::make_shared<Parameter>(element::u16, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 6, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("Coords' data type must be floating point"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_pooling_mode) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 6, spatial_scale, bin_not_used, bin_not_used, "invalid"),
                    NodeValidationFailure,
                    HasSubstr("Expected 'average' or 'bilinear' mode"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_features_rank) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 6, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("Expected a 4D tensor for the feature maps input"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_rois_rank) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 2});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 6, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("Expected a 2D tensor for the ROIs input with box coordinates"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_group_size) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 2});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 0, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("group_size has to be greater than 0"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_number_of_channels_and_group_size_in_avg_mode) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 2});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 2, 5, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("Number of input's channels must be a multiply of output_dim * group_size * group_size"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_output_dim_in_avg_mode) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 4, 2});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 17, 2, spatial_scale, bin_not_used, bin_not_used, "average"),
                    NodeValidationFailure,
                    HasSubstr("Number of input's channels must be a multiply of output_dim * group_size * group_size"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_spatial_bins_x) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 5, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 17, 2, spatial_scale, 0, 1, "bilinear"),
                    NodeValidationFailure,
                    HasSubstr("spatial_bins_x has to be greater than 0"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_spatial_bins_y) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 5, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(auto op = make_op(inputs, coords, 17, 2, spatial_scale, 1, 0, "bilinear"),
                    NodeValidationFailure,
                    HasSubstr("spatial_bins_y has to be greater than 0"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_number_of_channels_and_spatial_bins_in_bilinear_mode) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 5, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(
        auto op = make_op(inputs, coords, 17, 2, spatial_scale, 2, 5, "bilinear"),
        NodeValidationFailure,
        HasSubstr("Number of input's channels must be a multiply of output_dim * spatial_bins_x * spatial_bins_y"));
}

TEST_F(TypePropPSROIPoolingV0, invalid_output_dim_in_bilinear_mode) {
    const auto inputs = std::make_shared<Parameter>(element::f32, Shape{1, 72, 5, 5});
    const auto coords = std::make_shared<Parameter>(element::f32, Shape{150, 5});

    OV_EXPECT_THROW(
        auto op = make_op(inputs, coords, 10, 2, spatial_scale, 2, 4, "bilinear"),
        NodeValidationFailure,
        HasSubstr("Number of input's channels must be a multiply of output_dim * spatial_bins_x * spatial_bins_y"));
}

TEST_F(TypePropPSROIPoolingV0, features_dynamic_rank) {
    auto coords_shape = PartialShape{150, 5};
    auto symbols = set_shape_symbols(coords_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto coords = std::make_shared<Parameter>(element::f16, coords_shape);
    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, 0, 0, "average");

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({150, 2, 6, 6}));  // 4d
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropPSROIPoolingV0, rois_dynamic_rank) {
    auto feat_shape = PartialShape{1, 72, 4, 5};
    set_shape_symbols(feat_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, feat_shape);
    const auto coords = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, 0, 0, "average");

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 2, 6, 6}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropPSROIPoolingV0, dynamic_num_boxes) {
    auto coords_shape = PartialShape{{Dimension::dynamic(), 5}};
    auto symbols = set_shape_symbols(coords_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto coords = std::make_shared<Parameter>(element::f16, coords_shape);
    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, 0, 0, "average");

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 2, 6, 6}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropPSROIPoolingV0, feat_static_rank_dynamic_shape) {
    auto feat_shape = PartialShape::dynamic(4);
    auto coords_shape = PartialShape{{Dimension::dynamic(), 5}};
    set_shape_symbols(feat_shape);
    auto symbols = set_shape_symbols(coords_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, feat_shape);
    const auto coords = std::make_shared<Parameter>(element::f16, coords_shape);
    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, 0, 0, "average");

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 2, 6, 6}));  // 4d
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropPSROIPoolingV0, feat_and_rois_static_rank_dynamic_shape) {
    auto feat_shape = PartialShape::dynamic(4);
    auto coords_shape = PartialShape::dynamic(2);
    set_shape_symbols(feat_shape);
    auto symbols = set_shape_symbols(coords_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, feat_shape);
    const auto coords = std::make_shared<Parameter>(element::f16, coords_shape);
    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, 0, 0, "average");

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 2, 6, 6}));  // 4d
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropPSROIPoolingV0, feat_and_rois_interval_shapes) {
    auto feat_shape = PartialShape{{1, 2}, {10, 100}, {10, 20}, {30, 90}};
    auto coords_shape = PartialShape{{3, 10}, {1, 5}};
    set_shape_symbols(feat_shape);
    auto symbols = set_shape_symbols(coords_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, feat_shape);
    const auto coords = std::make_shared<Parameter>(element::f16, coords_shape);
    const auto op = make_op(inputs, coords, 2, 6, spatial_scale, 0, 0, "average");

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{3, 10}, 2, 6, 6}));  // 4d
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropPSROIPoolingV0, default_ctor) {
    auto feat_shape = PartialShape{2, {10, 100}, 10, 10};
    auto coords_shape = PartialShape{{3, 10}, {1, 5}};
    set_shape_symbols(feat_shape);
    auto symbols = set_shape_symbols(coords_shape);

    const auto inputs = std::make_shared<Parameter>(element::f16, feat_shape);
    const auto coords = std::make_shared<Parameter>(element::f16, coords_shape);

    const auto op = make_op();
    op->set_arguments(OutputVector{inputs, coords});
    op->set_output_dim(2);
    op->set_group_size(6);
    op->set_spatial_scale(spatial_scale);
    op->set_mode("average");
    op->validate_and_infer_types();

    EXPECT_FLOAT_EQ(op->get_spatial_scale(), spatial_scale);
    EXPECT_EQ(op->get_mode(), "average");
    EXPECT_EQ(op->get_group_size(), 6);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{3, 10}, 2, 6, 6}));  // 4d
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}
