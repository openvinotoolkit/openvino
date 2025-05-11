// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lrn.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, lrn_invalid_axes_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2});
    double alpha = 0.1, beta = 0.2, bias = 0.3;
    size_t size = 3;
    try {
        auto lrn = make_shared<op::v0::LRN>(data, axes, alpha, beta, bias, size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input axes must have rank equals 1"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    axes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});
    try {
        auto lrn = make_shared<op::v0::LRN>(data, axes, alpha, beta, bias, size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Number of elements of axes must be >= 0 and <= argument rank"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, lrn_incorrect_axes_value) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto axes = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{3, 4});
    double alpha = 0.1, beta = 0.2, bias = 0.3;
    size_t size = 3;
    try {
        auto lrn = make_shared<op::v0::LRN>(data, axes, alpha, beta, bias, size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis ("));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
