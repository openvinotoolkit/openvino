// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/string_tensor_unpack.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov {
namespace test {

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

TEST(type_prop, StringTensorUnpack_incorrect_data_type) {
    const auto data = std::make_shared<Parameter>(element::u8, PartialShape{3, 6});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::StringTensorUnpack>(data),
                    NodeValidationFailure,
                    HasSubstr("StringTensorUnpack expects a tensor with string elements"));
}
}  // namespace test
}  // namespace ov
