// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset11.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropROIPoolingV0 : public TypePropOpTest<op::v0::ROIPooling> {
protected:
    float spatial_scale = 0.625f;
    Shape pooling_roi_2x2{2, 2};
};

TEST_F(TypePropROIPoolingV0, default_ctor) {
    const auto feat_maps = make_shared<Parameter>(element::f32, PartialShape{{0, 3}, {1, 3}, {1, 6}, {1, 6}});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{{2, 4}, {1, 5}});

    const auto op = make_op();
    op->set_arguments(OutputVector{feat_maps, rois});
    op->set_spatial_scale(spatial_scale);
    op->set_method("max");
    op->set_output_roi({3, 4});
    op->validate_and_infer_types();

    EXPECT_FLOAT_EQ(op->get_spatial_scale(), spatial_scale);
    EXPECT_EQ(op->get_output_roi(), Shape({3, 4}));
    EXPECT_EQ(op->get_method(), "max");
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(static_cast<Node*>(op.get())->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{2, 4}, {1, 3}, 3, 4}));
}

TEST_F(TypePropROIPoolingV0, basic_shape_inference) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{1, 3, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, Shape{4, 5});
    const auto op = make_op(feat_maps, rois, pooling_roi_2x2, 0.625f);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_method(), "max");
    EXPECT_EQ(op->get_shape(), (Shape{4, 3, 2, 2}));
}

TEST_F(TypePropROIPoolingV0, dynamic_channels_dim) {
    auto feat_shape = PartialShape{1, -1, 6, 6};
    auto rois_shape = PartialShape{4, 5};
    auto feat_symbols = set_shape_symbols(feat_shape);
    auto rois_symbols = set_shape_symbols(rois_shape);

    const auto feat_maps = make_shared<Parameter>(element::f32, feat_shape);
    const auto rois = make_shared<Parameter>(element::f32, rois_shape);
    const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "max");

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, -1, 2, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(rois_symbols[0], feat_symbols[1], nullptr, nullptr));
}

TEST_F(TypePropROIPoolingV0, dynamic_num_rois_dim) {
    auto feat_shape = PartialShape{1, 3, 6, 6};
    auto rois_shape = PartialShape{-1, 5};
    auto feat_symbols = set_shape_symbols(feat_shape);
    auto rois_symbols = set_shape_symbols(rois_shape);

    const auto feat_maps = make_shared<Parameter>(element::f64, feat_shape);
    const auto rois = make_shared<Parameter>(element::f64, rois_shape);
    const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "bilinear");

    EXPECT_EQ(op->get_element_type(), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 3, 2, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(rois_symbols[0], feat_symbols[1], nullptr, nullptr));
}

TEST_F(TypePropROIPoolingV0, dynamic_rank_feat_maps) {
    const auto feat_maps = make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto rois = make_shared<Parameter>(element::f16, Shape{4, 5});
    const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale);

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, -1, 2, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropROIPoolingV0, dynamic_rank_feat_rois) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{1, 3, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 3, 2, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropROIPoolingV0, incompatible_input_rank) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{1, 3, 6, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "max"),
                    NodeValidationFailure,
                    HasSubstr("Expected a 4D tensor for the feature maps input"));
}

TEST_F(TypePropROIPoolingV0, incompatible_pooling_shape) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, Shape{2, 2, 2}, spatial_scale, "max"),
                    NodeValidationFailure,
                    HasSubstr("The dimension of pooled size is expected to be equal to 2"));
}

TEST_F(TypePropROIPoolingV0, incompatible_rois_second_dim) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 4});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "max"),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of ROIs input should contain batch id and box coordinates. This "
                              "dimension is expected to be equal to 5"));
}

TEST_F(TypePropROIPoolingV0, incompatible_feature_maps_element_type) {
    const auto feat_maps = make_shared<Parameter>(element::i32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "max"),
                    NodeValidationFailure,
                    HasSubstr("The data type for input and ROIs is expected to be a floating point type"));
}

TEST_F(TypePropROIPoolingV0, incompatible_rois_element_type) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::i16, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "bilinear"),
                    NodeValidationFailure,
                    HasSubstr("The data type for input and ROIs is expected to be a floating point type"));
}

TEST_F(TypePropROIPoolingV0, invalid_pooling_method) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, pooling_roi_2x2, spatial_scale, "invalid"),
                    NodeValidationFailure,
                    HasSubstr("Pooling method attribute should be either \'max\' or \'bilinear\'"));
}

TEST_F(TypePropROIPoolingV0, invalid_spatial_scale) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, pooling_roi_2x2, -1.0f),
                    NodeValidationFailure,
                    HasSubstr("The spatial scale attribute should be a positive floating point number"));
}

TEST_F(TypePropROIPoolingV0, invalid_pooled_size) {
    const auto feat_maps = make_shared<Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{3, 5});

    OV_EXPECT_THROW(const auto op = make_op(feat_maps, rois, Shape{1, 0}, spatial_scale),
                    NodeValidationFailure,
                    HasSubstr("Pooled size attributes pooled_h and pooled_w should should be positive integers"));
}
