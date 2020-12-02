//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "ngraph/op/detection_output.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

TEST(type_prop_layers, detection_output)
{
    auto box_logits = make_shared<op::Parameter>(element::f32, Shape{4, 25});
    auto class_preds = make_shared<op::Parameter>(element::f32, Shape{4, 20});
    auto proposals = make_shared<op::Parameter>(element::f32, Shape{4, 2, 20});
    auto aux_class_preds = make_shared<op::Parameter>(element::f32, Shape{4, 20});
    auto aux_box_preds = make_shared<op::Parameter>(element::f32, Shape{4, 25});
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 1;
    auto op = make_shared<op::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
}

TEST(type_prop_layers, detection_output_negative_keep_top_k)
{
    auto box_logits = make_shared<op::Parameter>(element::f32, Shape{4, 25});
    auto class_preds = make_shared<op::Parameter>(element::f32, Shape{4, 20});
    auto proposals = make_shared<op::Parameter>(element::f32, Shape{4, 2, 20});
    auto aux_class_preds = make_shared<op::Parameter>(element::f32, Shape{4, 20});
    auto aux_box_preds = make_shared<op::Parameter>(element::f32, Shape{4, 25});
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 1;
    auto op = make_shared<op::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 20, 7}));
}

TEST(type_prop_layers, detection_output_top_k)
{
    auto box_logits = make_shared<op::Parameter>(element::f32, Shape{4, 25});
    auto class_preds = make_shared<op::Parameter>(element::f32, Shape{4, 20});
    auto proposals = make_shared<op::Parameter>(element::f32, Shape{4, 2, 20});
    auto aux_class_preds = make_shared<op::Parameter>(element::f32, Shape{4, 20});
    auto aux_box_preds = make_shared<op::Parameter>(element::f32, Shape{4, 25});
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = 7;
    attrs.normalized = true;
    attrs.num_classes = 1;
    auto op = make_shared<op::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 28, 7}));
}

TEST(type_prop_layers, detection_output_all_dynamic_shapes)
{
    PartialShape dyn_shape = PartialShape::dynamic();
    auto box_logits = make_shared<op::Parameter>(element::f32, dyn_shape);
    auto class_preds = make_shared<op::Parameter>(element::f32, dyn_shape);
    auto proposals = make_shared<op::Parameter>(element::f32, dyn_shape);
    auto aux_class_preds = make_shared<op::Parameter>(element::f32, dyn_shape);
    auto aux_box_preds = make_shared<op::Parameter>(element::f32, dyn_shape);
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 1;
    auto op = make_shared<op::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 1, Dimension::dynamic(), 7}));
}

TEST(type_prop_layers, detection_output_dynamic_batch)
{
    auto box_logits =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 25});
    auto class_preds =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 20});
    auto proposals =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 20});
    auto aux_class_preds =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 20});
    auto aux_box_preds =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 25});
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 1;
    auto op = make_shared<op::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 1, Dimension::dynamic(), 7}}));
}
