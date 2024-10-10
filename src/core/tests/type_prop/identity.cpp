// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/Identity.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace testing;

class TypePropIdentityV15Test : public TypePropOpTest<ov::op::v15::Identity> {};

TEST_F(TypePropIdentityV15Test, default_ctor) {
    const auto data = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto op = make_op();
    op->set_arguments(ov::OutputVector{data});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), ov::element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({2, 2}));
}
