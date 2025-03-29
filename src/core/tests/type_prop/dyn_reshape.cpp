// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, reshape_v1_arg_rank_static_pattern_zero) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 0, 2, 8});
    auto pattern = ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 2, 0, 32});

    auto reshape_v1_static = make_shared<op::v1::Reshape>(arg, pattern, true);
    EXPECT_EQ(reshape_v1_static->get_output_shape(0), Shape({1, 2, 2, 32}));

    auto dynamic_arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto reshape_v1_dynamic = make_shared<op::v1::Reshape>(dynamic_arg, pattern, true);
    EXPECT_TRUE(
        reshape_v1_dynamic->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic(), 32}));
    try {
        auto static_shape_parameter = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto reshape_output_pattern = ov::op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 4});
        auto reshape = make_shared<op::v1::Reshape>(static_shape_parameter, reshape_output_pattern, true);
        FAIL() << "Expected failure on reshape construction";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("is incompatible with input shape"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
