// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

namespace {
using type = ngraph::element::Type;
void type_check(const type& refType) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(refType, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(refType, updates_shape);
    auto A = op::Constant::create(element::i32, Shape{1}, {1});
    auto scatter_update = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
    EXPECT_EQ(scatter_update->get_output_element_type(0), refType);
    EXPECT_EQ(scatter_update->get_output_shape(0), ref_shape);
}

void incorrect_type_check(const type& refType,
                          const type& indicesType,
                          const type& updatesType,
                          const type& axisType,
                          const std::string& errorStr) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(refType, ref_shape);
    auto I = make_shared<op::Parameter>(indicesType, indices_shape);
    auto U = make_shared<op::Parameter>(updatesType, updates_shape);
    auto A = op::Constant::create(axisType, Shape{1}, {1});
    try {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect element type of the input";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), errorStr);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

void incorrect_shape_check(const Shape& refShape,
                           const Shape& indicesShape,
                           const Shape& updatesShape,
                           const Shape& axisShape,
                           const float axisVal,
                           const std::string& errorStr) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::f32, refShape);
    auto I = make_shared<op::Parameter>(element::i32, indicesShape);
    auto U = make_shared<op::Parameter>(element::f32, updatesShape);
    auto A = op::Constant::create(element::i32, axisShape, {axisVal});
    try {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect shape of the input";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), errorStr);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
}  // namespace

TEST(type_prop, scatter_update_output_type_check_f16) {
    type_check(element::f16);
}

TEST(type_prop, scatter_update_output_type_check_f32) {
    type_check(element::f32);
}

TEST(type_prop, scatter_update_output_type_check_bf16) {
    type_check(element::bf16);
}

TEST(type_prop, scatter_update_output_type_check_i8) {
    type_check(element::i8);
}

TEST(type_prop, scatter_update_output_type_check_i16) {
    type_check(element::i16);
}

TEST(type_prop, scatter_update_output_type_check_i32) {
    type_check(element::i32);
}

TEST(type_prop, scatter_update_output_type_check_i64) {
    type_check(element::i64);
}

TEST(type_prop, scatter_update_output_type_check_u8) {
    type_check(element::u8);
}

TEST(type_prop, scatter_update_output_type_check_u16) {
    type_check(element::u16);
}

TEST(type_prop, scatter_update_output_type_check_u32) {
    type_check(element::u32);
}

TEST(type_prop, scatter_update_output_type_check_u64) {
    type_check(element::u64);
}

TEST(type_prop, scatter_update_v3_fail_updates_data_et_not_equal) {
    incorrect_type_check(element::f32,
                         element::i32,
                         element::u32,
                         element::i32,
                         "Element types for input data and updates do not match");
}

TEST(type_prop, scatter_update_v3_fail_indices_element_type) {
    incorrect_type_check(element::f32,
                         element::f16,
                         element::f32,
                         element::i64,
                         "Indices element type must be of an integral number type");
}

TEST(type_prop, scatter_update_v3_fail_axis_element_type) {
    incorrect_type_check(element::i16,
                         element::u64,
                         element::i16,
                         element::f32,
                         "Axis element type must be of an integral number type");
}

TEST(type_prop, scatter_update_v3_fail_updates_rank) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 1, 4},
                          {},
                          0,
                          "Updates rank is expected to be rank(indices) + rank(data) - 1");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_axis) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 2, 1, 4},
                          {},
                          0,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_indices) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 3, 1, 4},
                          {},
                          1,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_data_before_axis) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {3, 2, 1, 4},
                          {},
                          1,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_data_after_axis) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 2, 1, 5},
                          {},
                          1,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_axis_shape) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 2, 1, 4},
                          {2},
                          1,
                          "Axis input shape is required to be scalar or 1D tensor");
}

TEST(type_prop, scatter_update_v3_dynamic_data_shape) {
    PartialShape ref_shape = PartialShape::dynamic();
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::i8, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::i8, updates_shape);
    auto A = op::Constant::create(element::i16, Shape{}, {1});

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
    EXPECT_EQ(scatter_update->get_output_element_type(0), element::i8);
    EXPECT_TRUE(scatter_update->get_output_partial_shape(0).is_dynamic());
}
