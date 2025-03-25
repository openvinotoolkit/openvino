// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace opset9;
using namespace testing;

TEST(type_prop, grid_sample_default_constructor) {
    const auto data = make_shared<Parameter>(element::i32, PartialShape{1, 3, 4, 6});
    const auto grid = make_shared<Parameter>(element::f32, PartialShape{1, 7, 8, 2});
    auto op = make_shared<GridSample>();

    const auto& default_attrs = op->get_attributes();
    EXPECT_EQ(default_attrs.align_corners, false);
    EXPECT_EQ(default_attrs.mode, GridSample::InterpolationMode::BILINEAR);
    EXPECT_EQ(default_attrs.padding_mode, GridSample::PaddingMode::ZEROS);

    op->set_argument(0, data);
    op->set_argument(1, grid);

    op->set_attributes(
        GridSample::Attributes(true, GridSample::InterpolationMode::BICUBIC, GridSample::PaddingMode::BORDER));
    const auto& new_attrs = op->get_attributes();
    EXPECT_EQ(new_attrs.align_corners, true);
    EXPECT_EQ(new_attrs.mode, GridSample::InterpolationMode::BICUBIC);
    EXPECT_EQ(new_attrs.padding_mode, GridSample::PaddingMode::BORDER);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 3, 7, 8}));
}

TEST(type_prop, grid_sample_default_attributes) {
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

TEST(type_prop, grid_sample_interval_dims_and_labels) {
    auto data_pshape = PartialShape{{2, 4}, {1, 3}, 128, 256};
    auto data_symbols = set_shape_symbols(data_pshape);
    const auto data = make_shared<opset9::Parameter>(element::i32, data_pshape);

    auto grid_pshape = PartialShape{{3, 8}, {4, 6}, {5, 7}, 2};
    auto grid_symbols = set_shape_symbols(grid_pshape);
    const auto grid = make_shared<opset9::Parameter>(element::f32, grid_pshape);

    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    const auto& out_shape = grid_sample->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, (PartialShape{{3, 4}, {1, 3}, {4, 6}, {5, 7}}));
    EXPECT_THAT(get_shape_symbols(out_shape),
                ElementsAre(grid_symbols[0], data_symbols[1], grid_symbols[1], grid_symbols[2]));
}

TEST(type_prop, grid_sample_static_batch_data_labeled_dynamic_grid_batch) {
    auto data_pshape = PartialShape{2, {1, 3}, 224, 224};
    const auto data = make_shared<opset9::Parameter>(element::i32, data_pshape);

    auto grid_pshape = PartialShape{-1, {4, 6}, {5, 7}, 2};
    auto symbols = set_shape_symbols(grid_pshape);
    const auto grid = make_shared<opset9::Parameter>(element::f32, grid_pshape);

    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    const auto& out_shape = grid_sample->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, (PartialShape{2, {1, 3}, {4, 6}, {5, 7}}));
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], nullptr, symbols[1], symbols[2]));
}

TEST(type_prop, grid_sample_labeled_dynamic_batch_data_labeled_static_grid_batch) {
    auto data_pshape = PartialShape{-1, {1, 3}, 224, 224};
    auto symbols = set_shape_symbols(data_pshape);
    const auto data = make_shared<opset9::Parameter>(element::i32, data_pshape);

    auto grid_pshape = PartialShape{2, Dimension(4, 6), Dimension(5, 7), 2};
    const auto grid = make_shared<opset9::Parameter>(element::f32, grid_pshape);

    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    const auto& out_shape = grid_sample->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, (PartialShape{2, {1, 3}, {4, 6}, {5, 7}}));
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST(type_prop, grid_sample_labeled_interval_batch_data_dynamic_grid_batch) {
    auto data_pshape = PartialShape{{2, 4}, 3, 224, 224};
    auto symbols = set_shape_symbols(data_pshape);
    const auto data = make_shared<opset9::Parameter>(element::i32, data_pshape);

    auto grid_pshape = PartialShape{-1, 6, 7, 2};
    const auto grid = make_shared<opset9::Parameter>(element::f32, grid_pshape);

    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    const auto& out_shape = grid_sample->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, (PartialShape{{2, 4}, 3, 6, 7}));
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST(type_prop, grid_sample_dynamic_batch_data_labeled_interval_grid_batch) {
    auto data_pshape = PartialShape{-1, 3, 224, 224};
    const auto data = make_shared<opset9::Parameter>(element::i32, data_pshape);

    auto grid_pshape = PartialShape{{2, 4}, 6, 7, 2};
    auto symbols = set_shape_symbols(grid_pshape);
    const auto grid = make_shared<opset9::Parameter>(element::f32, grid_pshape);

    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    const auto& out_shape = grid_sample->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, (PartialShape{{2, 4}, 3, 6, 7}));
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], nullptr, symbols[1], symbols[2]));
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
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});
    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
        PartialShape{1, 3, Dimension::dynamic(), Dimension::dynamic()}))
        << "The output shape of GridSample is incorrect";
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

TEST(type_prop, grid_sample_dynamic_input_rank) {
    const auto data = make_shared<opset9::Parameter>(element::f16, PartialShape::dynamic());
    const auto grid = make_shared<opset9::Parameter>(element::f32, PartialShape{1, 5, 5, 2});
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});
    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(PartialShape{1, Dimension::dynamic(), 5, 5}))
        << "The output shape of GridSample is incorrect";
}

TEST(type_prop, grid_sample_dynamic_rank_of_data_and_grid) {
    const auto data = make_shared<opset9::Parameter>(element::f16, PartialShape::dynamic());
    const auto grid = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic());
    const auto grid_sample = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});
    EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}))
        << "The output shape of GridSample is incorrect";
}
