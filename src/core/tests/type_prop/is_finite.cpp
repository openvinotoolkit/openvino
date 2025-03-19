// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov::opset10;

TEST(type_prop, isfinite_output_shape) {
    const auto data = std::make_shared<Parameter>(ov::element::f16, ov::Shape{4, 2});
    const auto isfinite = std::make_shared<IsFinite>(data);

    EXPECT_EQ(isfinite->get_output_partial_shape(0), ov::PartialShape({4, 2}))
        << "The output shape of IsFinite is incorrect";
    ASSERT_EQ(isfinite->get_shape(), (ov::Shape{4, 2})) << "The output shape of IsFinite is incorrect";
}

TEST(type_prop, isfnite_sample_dynamic_batch) {
    const auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{ov::Dimension::dynamic(), 21, 37});
    const auto isfinite = std::make_shared<IsFinite>(data);

    EXPECT_EQ(isfinite->get_output_partial_shape(0), ov::PartialShape({ov::Dimension::dynamic(), 21, 37}))
        << "The output shape of IsFinite is incorrect";
}

TEST(type_prop, isfinite_output_type) {
    const auto data = std::make_shared<Parameter>(ov::element::f16, ov::Shape{4, 2});
    const auto isfinite = std::make_shared<IsFinite>(data);

    EXPECT_EQ(isfinite->get_element_type(), ov::element::boolean)
        << "The output element type of IsFinite is not a boolean.";
}

TEST(type_prop, isfinite_sample_interval_dimension) {
    const auto data = std::make_shared<Parameter>(ov::element::f16, (ov::PartialShape{ov::Dimension(2, 4), 73, 12}));
    const auto isfinite = std::make_shared<IsFinite>(data);

    EXPECT_EQ(isfinite->get_output_partial_shape(0), (ov::PartialShape{ov::Dimension(2, 4), 73, 12}))
        << "The output shape of IsFinnite is incorrect";
}

TEST(type_prop, isfinite_sample_dynamic_shape) {
    const auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape::dynamic(5));
    const auto isfinite = std::make_shared<IsFinite>(data);

    EXPECT_EQ(isfinite->get_output_partial_shape(0), (ov::PartialShape::dynamic(5)))
        << "The output shape of IsFinite is incorrect";
}

TEST(type_prop, isfinite_incorrect_input_type) {
    auto data = std::make_shared<Parameter>(ov::element::i64, ov::Shape{1, 2, 3});
    try {
        auto isfinite = std::make_shared<IsFinite>(data);
        FAIL() << "IsFinite invalid input type not detected";
    } catch (const ov::AssertFailure& error) {
        const auto exp_msg = "The element type of the input tensor must be a floating point number or dynamic";
        EXPECT_HAS_SUBSTRING(error.what(), exp_msg);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
