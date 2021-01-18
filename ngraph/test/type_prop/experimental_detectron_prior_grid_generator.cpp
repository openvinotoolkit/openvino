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

using Attrs = op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;
using GridGenerator = op::v6::ExperimentalDetectronPriorGridGenerator;

TEST(type_prop, detectron_grid_generator_static_shape_flatten)
{
    Attrs attrs;
    attrs.flatten = true;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    auto priors = std::make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto feature_map = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 200, 336});
    auto im_data = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 800, 1344});

    auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, attrs);

    ASSERT_EQ(grid_gen->get_output_element_type(0), element::f32);
    EXPECT_EQ(grid_gen->get_output_shape(0), (Shape{201600, 4}));
}

TEST(type_prop, detectron_grid_generator_static_shape_without_flatten)
{
    Attrs attrs;
    attrs.flatten = false;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    auto priors = std::make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto feature_map = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 200, 336});
    auto im_data = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 800, 1344});

    auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, attrs);

    ASSERT_EQ(grid_gen->get_output_element_type(0), element::f32);
    EXPECT_EQ(grid_gen->get_output_shape(0), (Shape{200, 336, 3, 4}));
}

TEST(type_prop, detectron_grid_generator_dynamic_shapes)
{
    Attrs attrs;
    attrs.flatten = false;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    struct ShapesAndAttrs
    {
        PartialShape priors_shape;
        PartialShape feature_map_shape;
        PartialShape ref_out_shape;
        bool flatten;
    };

    const Shape im_data_shape = Shape{1, 3, 800, 1344};
    const auto dyn_dim = Dimension::dynamic();

    std::vector<ShapesAndAttrs> shapes = {
        {{3, 4}, {1, 256, 200, dyn_dim}, {dyn_dim, 4}, true},
        {{3, 4}, {1, 256, dyn_dim, 336}, {dyn_dim, 4}, true},
        {{3, 4}, {1, 256, dyn_dim, dyn_dim}, {dyn_dim, 4}, true},
        {{dyn_dim, 4}, {1, 256, 200, dyn_dim}, {dyn_dim, 4}, true},
        {{dyn_dim, 4}, {1, 256, dyn_dim, 336}, {dyn_dim, 4}, true},
        {{dyn_dim, 4}, {1, 256, dyn_dim, dyn_dim}, {dyn_dim, 4}, true},
        {{3, 4}, {1, 256, 200, dyn_dim}, {200, dyn_dim, 3, 4}, false},
        {{3, 4}, {1, 256, dyn_dim, 336}, {dyn_dim, 336, 3, 4}, false},
        {{3, 4}, {1, 256, dyn_dim, dyn_dim}, {dyn_dim, dyn_dim, 3, 4}, false},
        {{dyn_dim, 4}, {1, 256, 200, dyn_dim}, {200, dyn_dim, dyn_dim, 4}, false},
        {{dyn_dim, 4}, {1, 256, dyn_dim, 336}, {dyn_dim, 336, dyn_dim, 4}, false},
        {{dyn_dim, 4}, {1, 256, dyn_dim, dyn_dim}, {dyn_dim, dyn_dim, dyn_dim, 4}, false}};

    for (const auto& s : shapes)
    {
        auto grid_attrs = attrs;
        grid_attrs.flatten = s.flatten;

        auto priors = std::make_shared<op::Parameter>(element::f32, s.priors_shape);
        auto feature_map = std::make_shared<op::Parameter>(element::f32, s.feature_map_shape);
        auto im_data = std::make_shared<op::Parameter>(element::f32, im_data_shape);

        auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, grid_attrs);

        ASSERT_EQ(grid_gen->get_output_element_type(0), element::f32);
        ASSERT_TRUE(grid_gen->get_output_partial_shape(0).same_scheme(s.ref_out_shape));
    }
}

TEST(type_prop, detectron_grid_generator_dynamic_shapes_intervals)
{
    Attrs attrs;
    attrs.flatten = false;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    struct ShapesAndAttrs
    {
        PartialShape priors_shape;
        PartialShape feature_map_shape;
        PartialShape ref_out_shape;
        bool flatten;
    };

    const Shape im_data_shape = Shape{1, 3, 800, 1344};

    std::vector<ShapesAndAttrs> shapes = {
        {{3, 4}, {1, 256, 200, Dimension(0, 100)}, {Dimension(0, 60000), 4}, true},
        {{3, 4}, {1, 256, Dimension(0, 150), 336}, {Dimension(0, 151200), 4}, true},
        {{3, 4}, {1, 256, Dimension(0, 150), Dimension(0, 100)}, {Dimension(0, 45000), 4}, true},
        {{Dimension(0, 3), 4}, {1, 256, 200, Dimension(0, 150)}, {Dimension(0, 90000), 4}, true},
        {{Dimension(0, 3), 4}, {1, 256, Dimension(0, 150), 336}, {Dimension(0, 151200), 4}, true},
        {{Dimension(0, 3), 4},
         {1, 256, Dimension(0, 150), Dimension(0, 100)},
         {Dimension(0, 45000), 4},
         true},
        {{3, 4}, {1, 256, 200, Dimension(0, 100)}, {200, Dimension(0, 100), 3, 4}, false},
        {{3, 4}, {1, 256, Dimension(0, 150), 336}, {Dimension(0, 150), 336, 3, 4}, false},
        {{3, 4},
         {1, 256, Dimension(0, 150), Dimension(0, 100)},
         {Dimension(0, 150), Dimension(0, 100), 3, 4},
         false},
        {{Dimension(0, 3), 4},
         {1, 256, 200, Dimension(0, 100)},
         {200, Dimension(0, 100), Dimension(0, 3), 4},
         false},
        {{Dimension(0, 3), 4},
         {1, 256, Dimension(0, 150), 336},
         {Dimension(0, 150), 336, Dimension(0, 3), 4},
         false},
        {{Dimension(0, 3), 4},
         {1, 256, Dimension(0, 150), Dimension(0, 100)},
         {Dimension(0, 150), Dimension(0, 100), Dimension(0, 3), 4},
         false}};

    for (const auto& s : shapes)
    {
        auto grid_attrs = attrs;
        grid_attrs.flatten = s.flatten;

        auto priors = std::make_shared<op::Parameter>(element::f32, s.priors_shape);
        auto feature_map = std::make_shared<op::Parameter>(element::f32, s.feature_map_shape);
        auto im_data = std::make_shared<op::Parameter>(element::f32, im_data_shape);

        auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, grid_attrs);

        ASSERT_EQ(grid_gen->get_output_element_type(0), element::f32);
        ASSERT_TRUE(grid_gen->get_output_partial_shape(0).same_scheme(s.ref_out_shape));
    }
}
