// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_not.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace ov;
using namespace testing;

using BitwiseNotTestParam = std::tuple<element::Type, PartialShape>;

namespace {
using namespace ov::element;
constexpr size_t exp_num_of_outputs = 1;

const auto types = Values(boolean, i8, i16, i32, i64, u8, u16, u32, u64);

const auto static_shapes = Values(PartialShape{0}, PartialShape{1}, PartialShape{2, 3, 7, 8});
const auto dynamic_shapes =
    Values(PartialShape::dynamic(3), PartialShape{2, {0, 5}, {4, -1}, -1, {3, 8}}, PartialShape::dynamic());
}  // namespace

class BitwiseNotTest : public TypePropOpTest<ov::op::v13::BitwiseNot>, public WithParamInterface<BitwiseNotTestParam> {
protected:
    void SetUp() override {
        std::tie(exp_type, exp_shape) = GetParam();
    }

    element::Type exp_type;
    PartialShape exp_shape;
};

INSTANTIATE_TEST_SUITE_P(type_prop_static_shape,
                         BitwiseNotTest,
                         Combine(types, static_shapes),
                         PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(type_prop_dynamic_shape,
                         BitwiseNotTest,
                         Combine(types, dynamic_shapes),
                         PrintToStringParamName());

TEST_P(BitwiseNotTest, propagate_dimensions) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(exp_type, exp_shape);
    const auto op = make_op(input);

    EXPECT_EQ(op->get_element_type(), exp_type);
    EXPECT_EQ(op->get_output_size(), exp_num_of_outputs);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}

TEST_P(BitwiseNotTest, propagate_symbols) {
    if (exp_shape.rank().is_static()) {
        set_shape_symbols(exp_shape);
    }
    const auto exp_symbols = get_shape_symbols(exp_shape);

    const auto input = std::make_shared<ov::op::v0::Parameter>(exp_type, exp_shape);
    const auto op = make_op(input);

    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), exp_symbols);
}

TEST_P(BitwiseNotTest, default_ctor) {
    const auto op = make_op();
    const auto input = std::make_shared<ov::op::v0::Parameter>(exp_type, exp_shape);

    op->set_argument(0, input);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), exp_type);
    EXPECT_EQ(op->get_output_size(), exp_num_of_outputs);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}

TEST(BitwiseNotTest, invalid_element_type) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v13::BitwiseNot>(data),
                    ov::NodeValidationFailure,
                    HasSubstr("The element type of the input tensor must be integer or boolean."));
}
