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
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{2});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2});

    op->set_arguments(ov::OutputVector{data, output_size, kernel_size});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, non_default_args) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{3, 12, 81});
    const auto output_size = std::make_shared<Parameter>(element::i64, Shape{2});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, Shape{2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{2, 2};
    const auto pads_end = Shape{2, 2};

    const auto op =
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{2, 2}));
    EXPECT_EQ(op->get_pads_end(), (Shape{2, 2}));
}

TEST_F(TypePropCol2ImTest, incorrect_types) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{2});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2});
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

TEST_F(TypePropCol2ImTest, batched_const_values) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{3, 12, 81});
    const auto output_size =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{16, 16});
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{2, 2};
    const auto pads_end = Shape{2, 2};

    const auto op =
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 3, 16, 16}));
}

TEST_F(TypePropCol2ImTest, unbatched_const_values) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{12, 324});
    const auto output_size =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{32, 32});
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op =
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 32, 32}));
}

TEST_F(TypePropCol2ImTest, incorrect_L) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{12, 325});
    const auto output_size =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{32, 32});
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    OV_EXPECT_THROW(std::ignore = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end),
                    ov::NodeValidationFailure,
                    HasSubstr("For given inputs and parameters the total number of data blocks must be equal to 324"));
}

TEST_F(TypePropCol2ImTest, incorrect_first_non_batch_dimension) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{13, 324});
    const auto output_size =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{32, 32});
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    OV_EXPECT_THROW(std::ignore = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end),
                    ov::NodeValidationFailure,
                    HasSubstr("First non-batch dimension is not evenly divisible by Product(kernel_shape)"));
}

TEST_F(TypePropCol2ImTest, dynamic_input_shapes) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op =
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, static_batch) {
    const auto data =
        std::make_shared<Parameter>(element::i64, PartialShape{5, Dimension::dynamic(), Dimension::dynamic()});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op =
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{5, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, 2D_dynamic_input) {
    const auto data =
        std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op =
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

}  // namespace test
}  // namespace ov
