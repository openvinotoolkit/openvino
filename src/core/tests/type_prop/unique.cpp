// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;

namespace {
void ASSERT_OUTPUT_SHAPES(const std::shared_ptr<ov::Node>& op, std::array<PartialShape, 4> expected_shapes) {
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(op->get_output_partial_shape(i).same_scheme(expected_shapes[i]))
            << "The output shape " << i << " of Unique is incorrect. Expected: " << expected_shapes[i]
            << ". Got: " << op->get_output_partial_shape(i);
    }
}

void ASSERT_ELEMENT_TYPES(const std::shared_ptr<ov::Node>& op, std::array<element::Type, 4> expected_types) {
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(op->get_output_element_type(i), expected_types[i])
            << "The output element type " << i << " of Unique is incorrect. Expected: " << expected_types[i]
            << ". Got: " << op->get_output_element_type(i);
    }
}
}  // namespace

TEST(type_prop, unique_no_axis_3d) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, 4, 2});
    const auto unique = make_shared<opset10::Unique>(data);

    ASSERT_ELEMENT_TYPES(unique, {element::f32, element::i64, element::i64, element::i64});
    ASSERT_OUTPUT_SHAPES(unique,
                         {PartialShape{{1, 16}}, PartialShape{{1, 16}}, PartialShape{{1, 16}}, PartialShape{{1, 16}}});
}

TEST(type_prop, unique_no_axis_3d_index_type_i32) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 3});
    const auto unique = make_shared<opset10::Unique>(data, true, element::i32);

    ASSERT_ELEMENT_TYPES(unique, {element::f32, element::i32, element::i32, element::i64});
    ASSERT_OUTPUT_SHAPES(unique,
                         {PartialShape{{1, 9}}, PartialShape{{1, 9}}, PartialShape{{1, 9}}, PartialShape{{1, 9}}});
}

TEST(type_prop, unique_no_axis_scalar) {
    const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{});
    const auto unique = make_shared<opset10::Unique>(data);

    ASSERT_ELEMENT_TYPES(unique, {element::i32, element::i64, element::i64, element::i64});
    ASSERT_OUTPUT_SHAPES(unique, {PartialShape{1}, PartialShape{1}, PartialShape{1}, PartialShape{1}});
}

TEST(type_prop, unique_no_axis_1D) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1});
    const auto unique = make_shared<opset10::Unique>(data);

    ASSERT_ELEMENT_TYPES(unique, {element::i64, element::i64, element::i64, element::i64});
    ASSERT_OUTPUT_SHAPES(unique, {PartialShape{1}, PartialShape{1}, PartialShape{1}, PartialShape{1}});
}

TEST(type_prop, unique_unsupported_index_et) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 3, 3});
    EXPECT_THROW(make_shared<opset10::Unique>(data, true, element::f32), ov::NodeValidationFailure);
}

// TEST(type_prop, grid_sample_dynamic_batch) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 3, 224, 224});
//     const auto grid = make_shared<opset10::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 10, 10, 2});
//     const auto grid_sample = make_shared<opset10::GridSample>(data, grid, opset10::GridSample::Attributes{});

//     EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 3, 10, 10}))
//         << "The output shape of GridSample is incorrect";
// }

// TEST(type_prop, grid_sample_dynamic_output_spatials) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{2, 3, 224, 224});
//     const auto grid =
//         make_shared<opset10::Parameter>(element::f64, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(),
//         2});
//     const auto grid_sample = make_shared<opset10::GridSample>(data, grid, opset10::GridSample::Attributes{});

//     EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
//         PartialShape{2, 3, Dimension::dynamic(), Dimension::dynamic()}))
//         << "The output shape of GridSample is incorrect";
// }

// TEST(type_prop, grid_sample_unsupported_grid_rank) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
//     const auto grid = make_shared<opset10::Parameter>(element::f64, PartialShape{1, 2, 3, 4, 5});
//     EXPECT_THROW(opset10::GridSample(data, grid, opset10::GridSample::Attributes{}), ov::NodeValidationFailure);
// }

// TEST(type_prop, grid_sample_unsupported_data_rank) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{1, 3, 224, 224, 224});
//     const auto grid = make_shared<opset10::Parameter>(element::f64, PartialShape{1, 5, 5, 2});
//     EXPECT_THROW(opset10::GridSample(data, grid, opset10::GridSample::Attributes{}), ov::NodeValidationFailure);
// }

// TEST(type_prop, grid_sample_unsupported_grid_element_type) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
//     const auto grid = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 5, 5, 2});
//     EXPECT_THROW(opset10::GridSample(data, grid, opset10::GridSample::Attributes{}), ov::NodeValidationFailure);
// }

// TEST(type_prop, grid_sample_incorrect_last_dim_in_grid) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
//     const auto grid = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 5, 5, 5});
//     EXPECT_THROW(opset10::GridSample(data, grid, opset10::GridSample::Attributes{}), ov::NodeValidationFailure);
// }

// TEST(type_prop, grid_sample_all_dimensions_dynamic_in_grid) {
//     const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{1, 3, 224, 224});
//     const auto grid = make_shared<opset10::Parameter>(
//         element::f32,
//         PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
//     const auto grid_sample = make_shared<opset10::GridSample>(data, grid, opset10::GridSample::Attributes{});
//     EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
//         PartialShape{1, 3, Dimension::dynamic(), Dimension::dynamic()}))
//         << "The output shape of GridSample is incorrect";
// }

// TEST(type_prop, grid_sample_all_dimensions_dynamic_in_data) {
//     const auto data = make_shared<opset10::Parameter>(
//         element::f16,
//         PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
//     const auto grid = make_shared<opset10::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, 5, 2});
//     const auto grid_sample = make_shared<opset10::GridSample>(data, grid, opset10::GridSample::Attributes{});
//     EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
//         PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 5}))
//         << "The output shape of GridSample is incorrect";
// }

// TEST(type_prop, grid_sample_dynamic_input_rank) {
//     const auto data = make_shared<opset10::Parameter>(element::f16, PartialShape::dynamic());
//     const auto grid = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 5, 5, 2});
//     const auto grid_sample = make_shared<opset10::GridSample>(data, grid, opset10::GridSample::Attributes{});
//     EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(PartialShape{1, Dimension::dynamic(), 5, 5}))
//         << "The output shape of GridSample is incorrect";
// }

// TEST(type_prop, grid_sample_dynamic_rank_of_data_and_grid) {
//     const auto data = make_shared<opset10::Parameter>(element::f16, PartialShape::dynamic());
//     const auto grid = make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic());
//     const auto grid_sample = make_shared<opset10::GridSample>(data, grid, opset10::GridSample::Attributes{});
//     EXPECT_TRUE(grid_sample->get_output_partial_shape(0).same_scheme(
//         PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}))
//         << "The output shape of GridSample is incorrect";
// }
