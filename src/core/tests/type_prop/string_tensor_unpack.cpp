// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/string_tensor_unpack.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov {
namespace test {

using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropStringTensorUnpackTest : public TypePropOpTest<op::v15::StringTensorUnpack> {};

TEST_F(TypePropStringTensorUnpackTest, 1D_static_input) {
    const auto data_shape = PartialShape{3};
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, 2D_static_input) {
    const auto data_shape = PartialShape{3, 9};
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, 3D_static_input) {
    const auto data_shape = PartialShape{3, 9, 1};
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, dynamic_data_rank) {
    const auto data = std::make_shared<Parameter>(element::string, PartialShape::dynamic());
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, interval_data_shape) {
    const auto data_shape = PartialShape{{4, 5}, {5, 6}};
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, single_interval_shape) {
    const auto data_shape = PartialShape{{4, 5}, 5};
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, single_dynamic_dim) {
    const auto data_shape = PartialShape{3, Dimension::dynamic()};
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = make_op(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropStringTensorUnpackTest, incorrect_data_type) {
    const auto data = std::make_shared<Parameter>(element::u8, PartialShape{3, 6});
    OV_EXPECT_THROW(std::ignore = make_op(data),
                    NodeValidationFailure,
                    HasSubstr("StringTensorUnpack expects a tensor with string elements"));
}
}  // namespace test
}  // namespace ov
