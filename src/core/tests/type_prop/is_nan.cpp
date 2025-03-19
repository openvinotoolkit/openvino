// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

TEST(type_prop, isnan_output_shape) {
    const auto data = make_shared<Parameter>(element::f16, Shape{4, 2});
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_output_partial_shape(0), (PartialShape{4, 2})) << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_sample_dynamic_batch) {
    const auto data = make_shared<Parameter>(element::f16, PartialShape{Dimension::dynamic(), 21, 37});
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 21, 37}))
        << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_sample_dynamic_shape) {
    const auto data = make_shared<Parameter>(element::f16, PartialShape::dynamic(5));
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_output_partial_shape(0), (PartialShape::dynamic(5)))
        << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_sample_dynamic_rank) {
    const auto data = make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_output_partial_shape(0), (PartialShape::dynamic()))
        << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_sample_interval_dimension) {
    const auto data = make_shared<Parameter>(element::f16, (PartialShape{Dimension(2, 4), 73, 12}));
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_output_partial_shape(0), (PartialShape{Dimension(2, 4), 73, 12}))
        << "The output shape of IsNaN is incorrect";
}

TEST(type_prop, isnan_output_type) {
    const auto data = make_shared<Parameter>(element::f16, Shape{4, 2});
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_element_type(), ov::element::boolean) << "The output element type of IsNaN is not a boolean.";
}

TEST(type_prop, isnan_bad_input_type) {
    auto data = make_shared<Parameter>(element::i64, Shape{1, 2, 3});
    try {
        auto isnan = make_shared<op::v10::IsNaN>(data);
        FAIL() << "IsNaN invalid input type not detected";
    } catch (const ov::AssertFailure& error) {
        const auto exp_msg = "The element type of the input tensor must be a floating point number.";
        EXPECT_HAS_SUBSTRING(error.what(), exp_msg);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, isnan_dynamic_input_type) {
    const auto data = make_shared<Parameter>(element::dynamic, Shape{3, 2, 1});
    const auto isnan = make_shared<IsNaN>(data);

    EXPECT_EQ(isnan->get_element_type(), ov::element::boolean) << "The output element type of IsNaN is not a boolean.";
}
