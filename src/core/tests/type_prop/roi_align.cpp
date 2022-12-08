// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop_layers, roi_align_basic_shape_inference) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{7, 4});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{7});
    const auto op = make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 2, 2, 1, 1.0f, "avg");
    ASSERT_EQ(op->get_shape(), (Shape{7, 3, 2, 2}));
}

TEST(type_prop_layers, roi_align_dynamic_channels_dim) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{10, Dimension(), 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{7, 4});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{7});
    const auto op = make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, "avg");
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{7, Dimension(), 3, 4}));
}

TEST(type_prop_layers, roi_align_num_rois_from_batch_indices) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape{Dimension{}, Dimension{}});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{9});
    const auto op = make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, "avg");
    ASSERT_EQ(op->get_shape(), (Shape{9, 3, 3, 4}));
}

TEST(type_prop_layers, roi_align_incompatible_num_rois) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension{}});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{2});
    // the first dimension of rois and batch_indices should be equal
    ASSERT_THROW(const auto unused = make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, "avg"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop_layers, roi_align_incompatible_input_rank) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{1, 4});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{1});
    // data rank needs to be 4
    ASSERT_THROW(const auto unused = make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, "avg"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop_layers, roi_align_incompatible_rois_second_dim) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{1, 5});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{1});
    // the second dim of rois needs to be 4
    ASSERT_THROW(const auto unused = make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, "avg"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop_layers, roi_align_9_basic_shape_inference) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{7, 4});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{7});
    const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum("avg");
    const auto aligned_mode = EnumNames<op::v9::ROIAlign::AlignedMode>::as_enum("half_pixel_for_nn");
    const auto op = make_shared<op::v9::ROIAlign>(data, rois, batch_indices, 2, 2, 1, 1.0f, pooling_mode, aligned_mode);
    ASSERT_EQ(op->get_shape(), (Shape{7, 3, 2, 2}));
}

TEST(type_prop_layers, roi_align_9_dynamic_channels_dim) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{10, Dimension(), 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{7, 4});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{7});
    const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum("avg");
    const auto aligned_mode = EnumNames<op::v9::ROIAlign::AlignedMode>::as_enum("half_pixel_for_nn");
    const auto op = make_shared<op::v9::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, pooling_mode, aligned_mode);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{7, Dimension(), 3, 4}));
}

TEST(type_prop_layers, roi_align_9_num_rois_from_batch_indices) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape{Dimension{}, Dimension{}});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{9});
    const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum("avg");
    const auto aligned_mode = EnumNames<op::v9::ROIAlign::AlignedMode>::as_enum("asymmetric");
    const auto op = make_shared<op::v9::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, pooling_mode, aligned_mode);
    ASSERT_EQ(op->get_shape(), (Shape{9, 3, 3, 4}));
}

TEST(type_prop_layers, roi_align_9_incompatible_num_rois) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension{}});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{2});
    const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum("avg");
    const auto aligned_mode = EnumNames<op::v9::ROIAlign::AlignedMode>::as_enum("asymmetric");
    // the first dimension of rois and batch_indices should be equal
    ASSERT_THROW(
        const auto unused =
            make_shared<op::v9::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, pooling_mode, aligned_mode),
        ngraph::NodeValidationFailure);
}

TEST(type_prop_layers, roi_align_9_default_aligned_mode) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{10, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension{}});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{2});
    const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum("avg");
    // the first dimension of rois and batch_indices should be equal
    ASSERT_THROW(
        const auto unused = make_shared<op::v9::ROIAlign>(data, rois, batch_indices, 3, 4, 1, 1.0f, pooling_mode),
        ngraph::NodeValidationFailure);
}
