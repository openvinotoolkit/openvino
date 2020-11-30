//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, roi_pooling_basic_shape_inference)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    const auto op = make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f);
    ASSERT_EQ(op->get_method(), "max");
    ASSERT_EQ(op->get_shape(), (Shape{4, 3, 2, 2}));
}

TEST(type_prop, roi_pooling_dynamic_channels_dim)
{
    const auto feat_maps =
        make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(), 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    const auto op = make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f, "max");
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{4, Dimension(), 2, 2}));
}

TEST(type_prop, roi_pooling_dynamic_num_rois_dim)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(), 5});
    const auto op = make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 3, 2, 2}));
}

TEST(type_prop, roi_pooling_dynamic_rank_feat_maps)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    const auto op = make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{4, Dimension(), 2, 2}));
}

TEST(type_prop, roi_pooling_dynamic_rank_rois)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 3, 2, 2}));
}

TEST(type_prop, roi_pooling_incompatible_input_rank)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    // feat_maps must be of rank 4
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f, "max"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_incompatible_pooling_shape)
{
    Shape pool_shape{2, 2, 2};
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    // pool_shape must be of rank 2 {pooled_h, pooled_w}
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, pool_shape, 0.625f, "max"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_incompatible_rois_second_dim)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    // the second dim of rois must be 5. [batch_id, x_1, y_1, x_2, y_2]
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f, "max"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_incompatible_feature_maps_element_type)
{
    const auto feat_maps = make_shared<op::Parameter>(element::i32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    // feat_maps element type must be floating point type
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f, "max"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_incompatible_rois_element_type)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f16, Shape{3, 5});
    // rois element type must be equal to feat_maps element type (floating point type)
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f, "bilinear"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_invalid_pooling_method)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f16, Shape{3, 5});
    // ROIPooling method is invalid: not max nor bilinear
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, 0.625f, "invalid"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_invalid_spatial_scale)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f16, Shape{3, 5});
    // ROIPooling spatial scale attribute must be a positive floating point number
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{2, 2}, -0.625f, "max"),
                 ngraph::NodeValidationFailure);
}

TEST(type_prop, roi_pooling_invalid_pooled_size)
{
    const auto feat_maps = make_shared<op::Parameter>(element::f32, Shape{3, 2, 6, 6});
    const auto rois = make_shared<op::Parameter>(element::f16, Shape{3, 5});
    // ROIPooling pooled_h and pooled_w must be non-negative integers
    ASSERT_THROW(make_shared<op::v0::ROIPooling>(feat_maps, rois, Shape{1, 0}, 0.625f, "max"),
                 ngraph::NodeValidationFailure);
}
