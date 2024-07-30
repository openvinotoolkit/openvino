// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/string_tensor_unpack.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {
namespace util {
size_t static get_character_count(const std::vector<std::string> text) {
    size_t total_char_count = 0;
    for (auto it = text.begin(); it != text.end(); it = std::next(it)) {
        total_char_count += it->size();
    }
    return total_char_count;
}
}  // namespace util

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropStringTensorUnpackTestSuite : public ::testing::TestWithParam<ov::PartialShape> {};

TEST_P(TypePropStringTensorUnpackTestSuite, TypePropStringTensorUnpackTestSuite) {
    const auto& data_shape = GetParam();
    const auto data = std::make_shared<Parameter>(element::string, data_shape);
    const auto op = std::make_shared<op::v15::StringTensorUnpack>(data);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(1), data_shape);
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape{Dimension::dynamic()});
}

INSTANTIATE_TEST_SUITE_P(TypePropStringTensorUnpackTestSuite,
                         TypePropStringTensorUnpackTestSuite,
                         ::testing::Values(PartialShape{3},
                                           PartialShape{3, 9},
                                           PartialShape{3, 9, 1},
                                           PartialShape::dynamic(),
                                           PartialShape{{4, 5}, {5, 6}},
                                           PartialShape{{4, 5}, 5},
                                           PartialShape{3, Dimension::dynamic()}));

class TypePropStringTensorUnpackTestSuiteWithConstant
    : public ::testing::TestWithParam<std::tuple<ov::Shape, std::vector<std::string>>> {};

TEST_P(TypePropStringTensorUnpackTestSuiteWithConstant, TypePropStringTensorUnpackTestSuiteWithConstant) {
    const auto& params = GetParam();
    const auto& shape = std::get<0>(params);
    const auto& text = std::get<1>(params);
    const auto data = std::make_shared<Constant>(element::string, shape, text);
    const auto op = std::make_shared<op::v15::StringTensorUnpack>(data);

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_shape(0), shape);
    EXPECT_EQ(op->get_output_shape(1), shape);
    EXPECT_EQ(op->get_output_shape(2), Shape{util::get_character_count(text)});
}

INSTANTIATE_TEST_SUITE_P(
    TypePropStringTensorUnpackTestSuiteWithConstant,
    TypePropStringTensorUnpackTestSuiteWithConstant,
    ::testing::Values(std::make_tuple(Shape{2}, std::vector<std::string>{"Intel", "OpenVINO"}),
                      std::make_tuple(Shape{1, 1, 2}, std::vector<std::string>{"Intel", "OpenVINO"}),
                      std::make_tuple(Shape{0}, std::vector<std::string>{}),
                      std::make_tuple(Shape{4}, std::vector<std::string>{"Intel", "OpenVINO", "", "AI"}),
                      std::make_tuple(Shape{2, 2}, std::vector<std::string>{"Intel", "OpenVINO", "Gen", "AI"})));

TEST(type_prop, StringTensorUnpack_incorrect_data_type) {
    const auto data = std::make_shared<Parameter>(element::u8, PartialShape{3, 6});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::StringTensorUnpack>(data),
                    NodeValidationFailure,
                    HasSubstr("StringTensorUnpack expects a tensor with string elements"));
}
}  // namespace test
}  // namespace ov
