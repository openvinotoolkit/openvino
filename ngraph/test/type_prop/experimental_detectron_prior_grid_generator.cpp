// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

struct GridGeneratorIntervalsTestParams
{
    PartialShape priors_shape;
    PartialShape feature_map_shape;
    PartialShape im_data_shape;
    PartialShape ref_out_shape;
    bool flatten;
};

struct GridGeneratorIntervalsTest : ::testing::TestWithParam<GridGeneratorIntervalsTestParams>
{
};

TEST_P(GridGeneratorIntervalsTest, detectron_grid_generator_dynamic_shapes_intervals_2)
{
    auto params = GetParam();

    Attrs attrs;
    attrs.flatten = false;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    auto grid_attrs = attrs;
    grid_attrs.flatten = params.flatten;

    auto priors = std::make_shared<op::Parameter>(element::f32, params.priors_shape);
    auto feature_map = std::make_shared<op::Parameter>(element::f32, params.feature_map_shape);
    auto im_data = std::make_shared<op::Parameter>(element::f32, params.im_data_shape);

    auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, grid_attrs);

    ASSERT_EQ(grid_gen->get_output_element_type(0), element::f32);
    ASSERT_TRUE(grid_gen->get_output_partial_shape(0).same_scheme(params.ref_out_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    GridGeneratorIntervalsTest,
    ::testing::Values(
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, 200, Dimension(0, 100)},
                                         {Dimension(0, 5), 3, 800, 1344},
                                         {Dimension(0, 60000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {Dimension(0, 7), 256, Dimension(0, 150), 336},
                                         {Dimension(0, 5), 3, 800, 1344},
                                         {Dimension(0, 151200), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, Dimension(0, 150), Dimension(0, 100)},
                                         {Dimension(0, 11), 3, 800, 1344},
                                         {Dimension(0, 45000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, 200, Dimension(0, 150)},
                                         {Dimension(0, 5), 3, 800, 1344},
                                         {Dimension(0, 90000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {Dimension(0, 77), 256, Dimension(0, 150), 336},
                                         {Dimension(0, 54), 3, 800, 1344},
                                         {Dimension(0, 151200), 4},
                                         true},
        GridGeneratorIntervalsTestParams{
            {Dimension(0, 3), 4},
            {Dimension(0, 3), 256, Dimension(0, 150), Dimension(0, 100)},
            {Dimension(0, 54), 3, 800, 1344},
            {Dimension(0, 45000), 4},
            true},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, 200, Dimension(0, 100)},
                                         {Dimension(0, 6), 3, 800, 1344},
                                         {200, Dimension(0, 100), 3, 4},
                                         false},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {Dimension(0, 9), 256, Dimension(0, 150), 336},
                                         {Dimension(0, 4), 3, 800, 1344},
                                         {Dimension(0, 150), 336, 3, 4},
                                         false},
        GridGeneratorIntervalsTestParams{
            {3, 4},
            {Dimension(1, 3), 256, Dimension(0, 150), Dimension(0, 100)},
            {Dimension(0, 4), 3, 800, 1344},
            {Dimension(0, 150), Dimension(0, 100), 3, 4},
            false},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {Dimension(5, 11), 256, 200, Dimension(0, 100)},
                                         {Dimension(0, 17), 3, 800, 1344},
                                         {200, Dimension(0, 100), Dimension(0, 3), 4},
                                         false},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {Dimension(7, 9), 256, Dimension(0, 150), 336},
                                         {Dimension(4, 18), 3, 800, 1344},
                                         {Dimension(0, 150), 336, Dimension(0, 3), 4},
                                         false},
        GridGeneratorIntervalsTestParams{
            {Dimension(0, 3), 4},
            {Dimension(0, 8), 256, Dimension(0, 150), Dimension(0, 100)},
            {Dimension(4, 18), 3, 800, 1344},
            {Dimension(0, 150), Dimension(0, 100), Dimension(0, 3), 4},
            false},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, 200, Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 60000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, Dimension(0, 150), 336},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 151200), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, Dimension(0, 150), Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 45000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, 200, Dimension(0, 150)},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 90000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, Dimension(0, 150), 336},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 151200), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, Dimension(0, 150), Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 45000), 4},
                                         true},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, 200, Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {200, Dimension(0, 100), 3, 4},
                                         false},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, Dimension(0, 150), 336},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 150), 336, 3, 4},
                                         false},
        GridGeneratorIntervalsTestParams{{3, 4},
                                         {1, 256, Dimension(0, 150), Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 150), Dimension(0, 100), 3, 4},
                                         false},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, 200, Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {200, Dimension(0, 100), Dimension(0, 3), 4},
                                         false},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, Dimension(0, 150), 336},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 150), 336, Dimension(0, 3), 4},
                                         false},
        GridGeneratorIntervalsTestParams{{Dimension(0, 3), 4},
                                         {1, 256, Dimension(0, 150), Dimension(0, 100)},
                                         Shape{1, 3, 800, 1344},
                                         {Dimension(0, 150), Dimension(0, 100), Dimension(0, 3), 4},
                                         false}),
    PrintToDummyParamName());
