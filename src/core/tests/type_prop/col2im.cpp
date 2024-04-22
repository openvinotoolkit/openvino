// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropCol2ImTest : public TypePropOpTest<op::v15::Col2Im> {};

TEST_F(TypePropCol2ImTest, default_ctor) {
    const auto op = make_op();
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{16, 16});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});

    op->set_arguments(ov::OutputVector{data, output_size, kernel_size});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 3, 16, 16}));
}

TEST_F(TypePropCol2ImTest, non_default_args) {
    const auto op = make_op();
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape{3, 12, 81});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{16, 16});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
    const auto strides = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
    const auto dilations = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
    const auto pads_begin = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
    const auto pads_end = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});

    op->set_arguments(ov::OutputVector{data, output_size, kernel_size, strides, dilations, pads_begin, pads_end});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 3, 16, 16}));
    //EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    //EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    //EXPECT_EQ(op->get_pads_begin(), (Shape{2, 2}));
    //EXPECT_EQ(op->get_pads_end(), (Shape{2, 2}));
}

TEST_F(TypePropCol2ImTest, incorrect_types) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{16, 16});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
    {
        const auto data_f32 = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 225});
        OV_EXPECT_THROW(std::ignore = make_op(data_f32, output_size, kernel_size),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the data tensor must be i32 or i64 type"));
    }
    {
        const auto output_size_i4 = std::make_shared<Parameter>(element::i4, PartialShape{16, 16});
        OV_EXPECT_THROW(std::ignore = make_op(data, output_size_i4, kernel_size),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the output_size tensor must be i32 or i64 type"));
    }
    {
        const auto kernel_size_u8 = std::make_shared<Parameter>(element::u8, PartialShape{2, 2});
        OV_EXPECT_THROW(std::ignore = make_op(data, output_size, kernel_size_u8),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the kernel_size tensor must be i32 or i64 type"));
    }
}

//TEST_F(TypePropCol2ImTest, incorrect_number_of_blocks) {
//    const auto op = make_op();
//    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
//    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{16, 16});
//    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
//
//    op->set_arguments(ov::OutputVector{data, output_size, kernel_size});
//    op->validate_and_infer_types();
//
//    EXPECT_EQ(op->get_output_size(), 1);
//    EXPECT_EQ(op->get_input_size(), 3);
//    EXPECT_EQ(op->get_output_element_type(0), element::i32);
//    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 3, 16, 16}));
//}

}  // namespace test
}  // namespace ov
