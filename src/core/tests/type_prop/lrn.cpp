// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, lrn_invalid_axes_rank) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    double alpha = 0.1, beta = 0.2, bias = 0.3;
    size_t size = 3;
    try {
        auto lrn = make_shared<op::LRN>(data, axes, alpha, beta, bias, size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input axes must have rank equals 1"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    axes = make_shared<op::Parameter>(element::f32, Shape{5});
    try {
        auto lrn = make_shared<op::LRN>(data, axes, alpha, beta, bias, size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Number of elements of axes must be >= 0 and <= argument rank"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, lrn_incorrect_axes_value) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{3, 4});
    double alpha = 0.1, beta = 0.2, bias = 0.3;
    size_t size = 3;
    try {
        auto lrn = make_shared<op::LRN>(data, axes, alpha, beta, bias, size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis ("));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
