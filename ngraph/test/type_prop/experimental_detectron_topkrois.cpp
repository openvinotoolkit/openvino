//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using ExperimentalTopKROIs = op::v6::ExperimentalDetectronTopKROIs;

TEST(type_prop, detectron_topk_rois)
{
    auto input_rois = std::make_shared<op::Parameter>(element::f32, Shape{5000, 4});
    auto rois_probs = std::make_shared<op::Parameter>(element::f32, Shape{5000});
    size_t max_rois = 1000;
    auto topk_rois = std::make_shared<ExperimentalTopKROIs>(input_rois, rois_probs, max_rois);

    ASSERT_EQ(topk_rois->get_output_element_type(0), element::f32);
    EXPECT_EQ(topk_rois->get_output_shape(0), (Shape{max_rois, 4}));
}

TEST(type_prop, detectron_topk_rois_dynamic)
{
    struct ShapesAndMaxROIs
    {
        PartialShape input_rois_shape;
        PartialShape rois_probs_shape;
        size_t max_rois;
    };

    const auto dyn_dim = Dimension::dynamic();
    const auto dyn_shape = PartialShape::dynamic();

    std::vector<ShapesAndMaxROIs> shapes_and_attrs = {{{3000, 4}, dyn_shape, 700},
                                                      {dyn_shape, {4000}, 600},
                                                      {dyn_shape, dyn_shape, 500},
                                                      {{dyn_dim, 4}, dyn_shape, 700},
                                                      {dyn_shape, {dyn_dim}, 600},
                                                      {dyn_shape, dyn_shape, 500},
                                                      {{dyn_dim, 4}, {dyn_dim}, 700}};

    for (const auto& s : shapes_and_attrs)
    {
        auto input_rois = std::make_shared<op::Parameter>(element::f32, s.input_rois_shape);
        auto rois_probs = std::make_shared<op::Parameter>(element::f32, s.rois_probs_shape);
        size_t max_rois = s.max_rois;
        auto topk_rois = std::make_shared<ExperimentalTopKROIs>(input_rois, rois_probs, max_rois);

        ASSERT_EQ(topk_rois->get_output_element_type(0), element::f32);
        EXPECT_EQ(topk_rois->get_output_shape(0), (Shape{max_rois, 4}));
    }
}
