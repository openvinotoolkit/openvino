// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grn.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, grn) {
    float bias = 1.25f;
    Shape data_shape{2, 3, 4, 5};
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto grn = make_shared<op::v0::GRN>(A, bias);

    EXPECT_EQ(grn->get_element_type(), element::f32);
    EXPECT_EQ(grn->get_shape(), data_shape);
}

TEST(type_prop, grn_dynamic) {
    float bias = 1.25f;
    PartialShape data_shape{2, Dimension::dynamic(), 3, Dimension(4, 6)};
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto grn = make_shared<op::v0::GRN>(A, bias);

    EXPECT_EQ(grn->get_element_type(), element::f32);
    EXPECT_EQ(grn->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, grn_invalid_data_rank) {
    float bias = 1.25f;
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    try {
        auto grn = make_shared<op::v0::GRN>(A, bias);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4, 5});

    try {
        auto grn = make_shared<op::v0::GRN>(A, bias);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
