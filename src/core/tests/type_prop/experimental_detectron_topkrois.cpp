// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using ExperimentalTopKROIs = op::v6::ExperimentalDetectronTopKROIs;

TEST(type_prop, detectron_topk_rois) {
    auto input_rois = std::make_shared<op::Parameter>(element::f32, Shape{5000, 4});
    auto rois_probs = std::make_shared<op::Parameter>(element::f32, Shape{5000});
    size_t max_rois = 1000;
    auto topk_rois = std::make_shared<ExperimentalTopKROIs>(input_rois, rois_probs, max_rois);

    ASSERT_EQ(topk_rois->get_output_element_type(0), element::f32);
    EXPECT_EQ(topk_rois->get_output_shape(0), (Shape{max_rois, 4}));
}

TEST(type_prop, detectron_topk_rois_dynamic) {
    struct ShapesAndMaxROIs {
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

    for (const auto& s : shapes_and_attrs) {
        auto input_rois = std::make_shared<op::Parameter>(element::f32, s.input_rois_shape);
        auto rois_probs = std::make_shared<op::Parameter>(element::f32, s.rois_probs_shape);
        size_t max_rois = s.max_rois;
        auto topk_rois = std::make_shared<ExperimentalTopKROIs>(input_rois, rois_probs, max_rois);

        ASSERT_EQ(topk_rois->get_output_element_type(0), element::f32);
        EXPECT_EQ(topk_rois->get_output_shape(0), (Shape{max_rois, 4}));
    }
}
