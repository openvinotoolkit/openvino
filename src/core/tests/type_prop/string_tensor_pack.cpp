// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/string_tensor_pack.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov {
namespace test {

using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropStringTensorPackTest : public TypePropOpTest<op::v15::StringTensorPack> {};

TEST_F(TypePropStringTensorPackTest, 1D_static_input) {
    const auto indices_shape = PartialShape{3};
    const auto begins = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto ends = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = make_op(begins, ends, symbols);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), indices_shape);
}

TEST_F(TypePropStringTensorPackTest, 2D_static_input) {
    const auto indices_shape = PartialShape{3, 6};
    const auto begins = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto ends = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = make_op(begins, ends, symbols);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), indices_shape);
}

TEST_F(TypePropStringTensorPackTest, fp64_indices) {
    const auto indices_shape = PartialShape{3};
    const auto begins = std::make_shared<Parameter>(element::i64, indices_shape);
    const auto ends = std::make_shared<Parameter>(element::i64, indices_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = make_op(begins, ends, symbols);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), indices_shape);
}

TEST_F(TypePropStringTensorPackTest, dynamic_indices_shape) {
    const auto indices_shape = PartialShape::dynamic();
    const auto begins = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto ends = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = make_op(begins, ends, symbols);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), indices_shape);
}

//
//TEST_F(TypePropStringTensorPackTest, interval_data_shape) {
//    const auto data_shape = PartialShape{{4, 5}, {5, 6}};
//    const auto data = std::make_shared<Parameter>(element::string, data_shape);
//    const auto op = make_op(data);
//    op->validate_and_infer_types();
//
//    EXPECT_EQ(op->get_output_element_type(0), element::i32);
//    EXPECT_EQ(op->get_output_element_type(1), element::i32);
//    EXPECT_EQ(op->get_output_element_type(2), element::u8);
//    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
//    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
//    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
//}
//
//TEST_F(TypePropStringTensorPackTest, single_interval_shape) {
//    const auto data_shape = PartialShape{{4, 5}, 5};
//    const auto data = std::make_shared<Parameter>(element::string, data_shape);
//    const auto op = make_op(data);
//    op->validate_and_infer_types();
//
//    EXPECT_EQ(op->get_output_element_type(0), element::i32);
//    EXPECT_EQ(op->get_output_element_type(1), element::i32);
//    EXPECT_EQ(op->get_output_element_type(2), element::u8);
//    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
//    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
//    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
//}
//
//TEST_F(TypePropStringTensorPackTest, single_dynamic_dim) {
//    const auto data_shape = PartialShape{3, Dimension::dynamic()};
//    const auto data = std::make_shared<Parameter>(element::string, data_shape);
//    const auto op = make_op(data);
//    op->validate_and_infer_types();
//
//    EXPECT_EQ(op->get_output_element_type(0), element::i32);
//    EXPECT_EQ(op->get_output_element_type(1), element::i32);
//    EXPECT_EQ(op->get_output_element_type(2), element::u8);
//    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
//    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
//    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
//}
//
//TEST_F(TypePropStringTensorPackTest, incorrect_data_type) {
//    const auto data = std::make_shared<Parameter>(element::u8, PartialShape{3, 6});
//    OV_EXPECT_THROW(std::ignore = make_op(data),
//                    NodeValidationFailure,
//                    HasSubstr("StringTensorUnpack expects a tensor with string elements"));
//}
}  // namespace test
}  // namespace ov
