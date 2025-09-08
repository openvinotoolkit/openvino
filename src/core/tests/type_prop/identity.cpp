// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace testing;

namespace ov {
namespace test {

class TypePropIdentityV16Test : public TypePropOpTest<op::v16::Identity> {};

TEST_F(TypePropIdentityV16Test, default_ctor) {
    const auto data = op::v0::Constant::create(element::f64, Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto op = make_op();
    op->set_arguments(OutputVector{data});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 2}));
}

TEST_F(TypePropIdentityV16Test, input_data_ctor) {
    const auto data = op::v0::Constant::create(element::i64, Shape{1, 2}, {1.0f, 1.0f});
    const auto op = make_op(data);

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 2}));
}
}  // namespace test
}  // namespace ov