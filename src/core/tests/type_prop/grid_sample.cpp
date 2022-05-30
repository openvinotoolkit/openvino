// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/opsets/opset9.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, grid_sample_default) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(element::f32, PartialShape{1, 10, 10, 2});
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    EXPECT_EQ(grid_sample->get_element_type(), data->get_element_type())
        << "The output element type of GridSample doesn't match the input data element type";
    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(PartialShape{1, 3, 10, 10}))
        << "The output shape of GridSample is incorrect";
}

TEST(type_prop, grid_sample_dynamic_batch) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 3, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 10, 10, 2});
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 3, 10, 10}))
        << "The output shape of GridSample is incorrect";
}

TEST(type_prop, grid_sample_dynamic_output_spatials) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{2, 3, 224, 224});
    const auto grid =
        make_shared<opset9::Parameter>(element::f64, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 2});
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
        PartialShape{2, 3, Dimension::dynamic(), Dimension::dynamic()}))
        << "The output shape of GridSample is incorrect";
}

TEST(type_prop, grid_sample_unsupported_grid_rank) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(element::f64, PartialShape{1, 2, 3, 4, 5});
    EXPECT_THROW(opset9::GridSample(data, grid, opset9::GridSample::Attributes{}), ov::NodeValidationFailure);
}

TEST(type_prop, grid_sample_unsupported_data_rank) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{1, 3, 224, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(element::f64, PartialShape{1, 5, 5, 2});
    EXPECT_THROW(opset9::GridSample(data, grid, opset9::GridSample::Attributes{}), ov::NodeValidationFailure);
}

TEST(type_prop, grid_sample_unsupported_grid_element_type) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(element::i64, PartialShape{1, 5, 5, 2});
    EXPECT_THROW(opset9::GridSample(data, grid, opset9::GridSample::Attributes{}), ov::NodeValidationFailure);
}

TEST(type_prop, grid_sample_incorrect_last_dim_in_grid) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(element::f32, PartialShape{1, 5, 5, 5});
    EXPECT_THROW(opset9::GridSample(data, grid, opset9::GridSample::Attributes{}), ov::NodeValidationFailure);
}

TEST(type_prop, grid_sample_all_dimensions_dynamic_in_grid) {
    const auto data = make_shared<opset9::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
    const auto grid = make_shared<opset9::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    EXPECT_THROW(opset9::GridSample(data, grid, opset9::GridSample::Attributes{}), ov::NodeValidationFailure);
}

TEST(type_prop, grid_sample_all_dimensions_dynamic_in_data) {
    const auto data = make_shared<opset9::Parameter>(
        element::f16,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    const auto grid = make_shared<opset9::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, 5, 2});
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});
    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 5}))
        << "The output shape of GridSample is incorrect";
}
