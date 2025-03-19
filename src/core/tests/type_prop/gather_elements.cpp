// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/ops.hpp"

using namespace std;
using namespace ov;
using namespace op;
using namespace testing;

// ------------------------------ V6 ------------------------------

TEST(type_prop, gather_elements_default_constructor) {
    PartialShape data_shape{1, 2, 3, 4};
    PartialShape indices_shape{1, 2, 10, 4};
    int64_t axis = -2;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>();

    op->set_axis(axis);
    EXPECT_EQ(op->get_axis(), axis);

    op->set_argument(0, data);
    op->set_argument(1, indices);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), indices_shape);
}

TEST(type_prop, gather_elements_2D_axis_0) {
    Shape data_shape{3, 3};
    Shape indices_shape{2, 3};
    int axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(op->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_2D_axis_1) {
    Shape data_shape{3, 3};
    Shape indices_shape{3, 1};
    int axis = 1;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(op->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_3D_axis_0) {
    Shape data_shape{3, 3, 10000};
    Shape indices_shape{300, 3, 10000};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(op->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_3D_axis_2) {
    Shape data_shape{300, 3, 10};
    Shape indices_shape{300, 3, 10000};
    int64_t axis = 2;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(op->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_4D_axis_minus_1) {
    Shape data_shape{300, 3, 10, 1};
    Shape indices_shape{300, 3, 10, 33333};
    int64_t axis = -1;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(op->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_nonfloat_data_type_int64_indices) {
    Shape data_shape{300, 3, 10, 1};
    Shape indices_shape{300, 3, 10, 33333};
    int64_t axis = -1;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_dynamic_consistent_shapes) {
    PartialShape data_shape{4, 4, Dimension::dynamic()};
    PartialShape indices_shape{1, Dimension::dynamic(), 5};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_shape(), Shape({1, 4, 5}));
}

TEST(type_prop, gather_elements_dynamic_out_shape) {
    PartialShape data_shape{4, 4, Dimension::dynamic()};
    PartialShape indices_shape{1, Dimension::dynamic(), Dimension::dynamic()};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 4, Dimension::dynamic()}));
}

TEST(type_prop, gather_elements_interval_shapes) {
    PartialShape data_shape{4, Dimension(1, 7), 5};
    PartialShape indices_shape{1, Dimension(5, 10), 5};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, {5, 7}, 5}));
}

TEST(type_prop, gather_elements_data_rank_dynamic_indices_rank_static) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape indices_shape{4, 7, 5};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({4, 7, 5}));
}

TEST(type_prop, gather_elements_data_rank_static_indices_rank_dynamic) {
    PartialShape data_shape{4, Dimension(1, 7), 5};
    PartialShape indices_shape = PartialShape::dynamic();
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), {1, 7}, 5}));
}

TEST(type_prop, gather_elements_data_pshape_static_indices_rank_dynamic) {
    PartialShape data_shape{4, 7, 5};
    PartialShape indices_shape = PartialShape::dynamic();
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);
    EXPECT_EQ(op->get_element_type(), element::Type_t::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 7, 5}));
}

TEST(type_prop, gather_elements_interval_dims_with_symbols_both_inputs) {
    PartialShape data_shape{-1, {2, 4}, {1, 5}, -1, {4, 8}, {2, 4}};
    auto symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{-1, {3, 6}, {6, 10}, {4, 8}, -1, {4, 6}};
    auto ind_symbols = set_shape_symbols(indices_shape);

    int64_t axis = 2;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(out_shape, PartialShape({-1, {3, 4}, {6, 10}, {4, 8}, {4, 8}, 4}));
    EXPECT_THAT(get_shape_symbols(out_shape),
                ElementsAre(symbols[0], symbols[1], ind_symbols[2], symbols[3], symbols[4], symbols[5]));
}

TEST(type_prop, gather_elements_interval_dims_with_symbols_data) {
    PartialShape data_shape{-1, {2, 4}, {1, 5}, -1, {4, 8}, {2, 4}};
    auto symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{-1, {3, 6}, {6, 10}, {4, 8}, -1, {4, 6}};

    int64_t axis = 2;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(out_shape, PartialShape({-1, {3, 4}, {6, 10}, {4, 8}, {4, 8}, 4}));
    EXPECT_THAT(get_shape_symbols(out_shape),
                ElementsAre(symbols[0], symbols[1], nullptr, symbols[3], symbols[4], symbols[5]));
}

TEST(type_prop, gather_elements_interval_dims_with_symbols_indices) {
    PartialShape data_shape{-1, {2, 4}, {1, 5}, -1, {4, 8}, {2, 4}};
    PartialShape indices_shape{-1, {3, 6}, {6, 10}, {4, 8}, -1, {4, 6}};
    auto symbols = set_shape_symbols(indices_shape);

    int64_t axis = 2;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);
    auto op = make_shared<v6::GatherElements>(data, indices, axis);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(out_shape, PartialShape({-1, {3, 4}, {6, 10}, {4, 8}, {4, 8}, 4}));
    EXPECT_THAT(get_shape_symbols(out_shape), symbols);
}
// --------------------- Negative tests ------------------------------

TEST(type_prop, gather_elements_type_inconsistency) {
    Shape data_shape{3, 3};
    Shape indices_shape{2, 1};
    int64_t axis = 1;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::u32, indices_shape);

    try {
        auto op = make_shared<v6::GatherElements>(data, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "the indices tensor type check failed";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("indices must be of int32 or int64 type. But instead got"));
    } catch (...) {
        FAIL() << "type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_out_of_bounds_axis) {
    Shape data_shape{3, 3};
    Shape indices_shape{2, 1};
    int64_t axis = -33;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);

    try {
        auto op = make_shared<v6::GatherElements>(data, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "axis out of bounds check failed";
    } catch (const ov::AssertFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("out of the tensor rank range"));
    } catch (...) {
        FAIL() << "axis out of bounds check failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_rank_consistency_check) {
    Shape data_shape{3, 3};
    Shape indices_shape{2, 3, 3333};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);

    try {
        auto op = make_shared<v6::GatherElements>(data, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "rank consistency check failed";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("data and indices rank must be equal"));
    } catch (...) {
        FAIL() << "rank consistency check failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_shape_inconsistency) {
    Shape data_shape{3, 3};
    Shape indices_shape{2, 1};
    int64_t axis = 1;
    const auto data = make_shared<v0::Parameter>(element::Type_t::f32, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i32, indices_shape);

    try {
        auto op = make_shared<v6::GatherElements>(data, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Shape inconsistency check failed";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("are not consistent"));
    } catch (...) {
        FAIL() << "Shape inconsistency check failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_dynamic_inconsistent_shapes) {
    PartialShape data_shape{4, 2, 4, Dimension::dynamic()};
    PartialShape indices_shape{1, 3, Dimension::dynamic(), 5};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);

    try {
        auto op = make_shared<v6::GatherElements>(data, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Shape inconsistency check for dynamic PartialShape failed";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("are not consistent"));
    } catch (...) {
        FAIL() << "Shape inconsistency check for dynamic PartialShape failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_incosistent_interval_shapes) {
    PartialShape data_shape{4, 4, 5};
    PartialShape indices_shape{1, Dimension(5, 10), 5};
    int64_t axis = 0;
    const auto data = make_shared<v0::Parameter>(element::Type_t::i8, data_shape);
    const auto indices = make_shared<v0::Parameter>(element::Type_t::i64, indices_shape);

    try {
        auto op = make_shared<v6::GatherElements>(data, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Shape inconsistency check for dynamic PartialShape failed";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("are not consistent"));
    } catch (...) {
        FAIL() << "Shape inconsistency check for dynamic PartialShape failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_data_rank_dynamic_indicies_incorrect_rank) {
    constexpr int64_t axis = 0;
    const auto data = std::make_shared<v0::Parameter>(element::i8, PartialShape::dynamic());
    const auto indices = std::make_shared<v0::Parameter>(element::i64, PartialShape{});

    OV_EXPECT_THROW(std::ignore = std::make_shared<v6::GatherElements>(data, indices, axis),
                    NodeValidationFailure,
                    HasSubstr("indices rank must be >= 1"));
}
