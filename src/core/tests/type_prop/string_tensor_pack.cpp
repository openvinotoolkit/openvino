// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/string_tensor_pack.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov {
namespace test {

using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropStringTensorPackTestSuite : public ::testing::TestWithParam<ov::PartialShape> {};

TEST_P(TypePropStringTensorPackTestSuite, TypePropStringTensorPackTestSuite) {
    const auto& indices_shape = GetParam();
    const auto begins = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto ends = std::make_shared<Parameter>(element::i32, indices_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = std::make_shared<op::v15::StringTensorPack>(begins, ends, symbols);

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), indices_shape);
}

INSTANTIATE_TEST_SUITE_P(TypePropStringTensorPackTestSuite,
                         TypePropStringTensorPackTestSuite,
                         ::testing::Values(PartialShape{3},
                                           PartialShape{3, 9},
                                           PartialShape{3, 9, 1},
                                           PartialShape::dynamic(),
                                           PartialShape{{4, 5}, {5, 6}},
                                           PartialShape{{4, 5}, 5},
                                           PartialShape{3, Dimension::dynamic()}));

using TypePropStringTensorPackV15Test = TypePropOpTest<op::v15::StringTensorPack>;

TEST_F(TypePropStringTensorPackV15Test, begins_static_ends_dynamic) {
    auto begins_shape = PartialShape{3};
    auto ends_shape = PartialShape{Dimension::dynamic()};
    auto begins_symbols = set_shape_symbols(begins_shape);
    auto ends_symbols = set_shape_symbols(ends_shape);
    const auto begins = std::make_shared<Parameter>(element::i32, begins_shape);
    const auto ends = std::make_shared<Parameter>(element::i32, ends_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{Dimension::dynamic()});
    const auto op = make_op(begins, ends, symbols);

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), begins_shape);
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), begins_symbols);
}

TEST_F(TypePropStringTensorPackV15Test, ends_static_begins_dynamic) {
    auto begins_shape = PartialShape{Dimension::dynamic()};
    auto ends_shape = PartialShape{5};
    auto begins_symbols = set_shape_symbols(begins_shape);
    auto ends_symbols = set_shape_symbols(ends_shape);
    const auto begins = std::make_shared<Parameter>(element::i32, begins_shape);
    const auto ends = std::make_shared<Parameter>(element::i32, ends_shape);
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{Dimension::dynamic()});
    const auto op = make_op(begins, ends, symbols);

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), ends_shape);
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), begins_symbols);
}

TEST_F(TypePropStringTensorPackV15Test, default_case) {
    const auto begins = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = make_op(begins, ends, symbols);

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape{3});
}

TEST_F(TypePropStringTensorPackV15Test, int64_indices) {
    const auto begins = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto ends = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    const auto op = make_op(begins, ends, symbols);

    EXPECT_EQ(op->get_output_element_type(0), element::string);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape{3});
}

TEST_F(TypePropStringTensorPackV15Test, incorrect_symbols_shape) {
    const auto begins = std::make_shared<Parameter>(element::i32, PartialShape{2});
    const auto ends = std::make_shared<Parameter>(element::i32, PartialShape{2});
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100, 3});
    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols),
                    NodeValidationFailure,
                    HasSubstr("Symbols input must be 1D."));
}

TEST_F(TypePropStringTensorPackV15Test, begins_ends_shape_mismatch) {
    const auto begins = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<Parameter>(element::i32, PartialShape{2});
    const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols),
                    NodeValidationFailure,
                    HasSubstr("The shapes of begins and ends have to be compatible"));
}

TEST_F(TypePropStringTensorPackV15Test, incorrect_types) {
    {
        const auto begins = std::make_shared<Parameter>(element::i8, PartialShape{3});
        const auto ends = std::make_shared<Parameter>(element::i8, PartialShape{3});
        const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
        OV_EXPECT_THROW(
            std::ignore = make_op(begins, ends, symbols),
            NodeValidationFailure,
            HasSubstr("The element types of the begins and ends input tensors must match and be of i32 or i64 type"));
    }
    {
        const auto begins = std::make_shared<Parameter>(element::i32, PartialShape{3});
        const auto ends = std::make_shared<Parameter>(element::i64, PartialShape{3});
        const auto symbols = std::make_shared<Parameter>(element::u8, PartialShape{100});
        OV_EXPECT_THROW(
            std::ignore = make_op(begins, ends, symbols),
            NodeValidationFailure,
            HasSubstr("The element types of the begins and ends input tensors must match and be of i32 or i64 type"));
    }
    {
        const auto begins = std::make_shared<Parameter>(element::i32, PartialShape{3});
        const auto ends = std::make_shared<Parameter>(element::i32, PartialShape{3});
        const auto symbols = std::make_shared<Parameter>(element::f32, PartialShape{100});
        OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols),
                        NodeValidationFailure,
                        HasSubstr("StringTensorPack expects a tensor with ov::element::u8 elements"));
    }
}
}  // namespace test
}  // namespace ov
