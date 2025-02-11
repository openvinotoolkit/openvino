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
using ov::op::v3::ShapeOf;
using testing::HasSubstr;

class TypePropCol2ImTest : public TypePropOpTest<op::v15::Col2Im> {};

TEST_F(TypePropCol2ImTest, default_ctor) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{2});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2});

    const auto op = make_op(data, output_size, kernel_size);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(op->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{0, 0}));
    EXPECT_EQ(op->get_pads_end(), (Shape{0, 0}));
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, non_default_args) {
    PartialShape data_shape{3, 12, 81};
    auto data_symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::i64, data_shape);
    const auto output_size = std::make_shared<Parameter>(element::i64, Shape{2});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, Shape{2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{2, 2};
    const auto pads_end = Shape{2, 2};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{2, 2}));
    EXPECT_EQ(op->get_pads_end(), (Shape{2, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(data_symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropCol2ImTest, incorrect_types) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape{2});
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape{2});
    constexpr auto error_substring =
        "The element types of the output_size and kernel_size tensors must match and be of i32 or i64 type";
    {
        const auto output_size_i4 = std::make_shared<Parameter>(element::i4, PartialShape{16, 16});
        OV_EXPECT_THROW(std::ignore = make_op(data, output_size_i4, kernel_size),
                        ov::NodeValidationFailure,
                        HasSubstr(error_substring));
    }
    {
        const auto kernel_size_u8 = std::make_shared<Parameter>(element::u8, PartialShape{2, 2});
        OV_EXPECT_THROW(std::ignore = make_op(data, output_size, kernel_size_u8),
                        ov::NodeValidationFailure,
                        HasSubstr(error_substring));
    }
    {
        const auto output_size_i32 = std::make_shared<Parameter>(element::i32, PartialShape{16, 16});
        const auto kernel_size_i64 = std::make_shared<Parameter>(element::i64, PartialShape{2, 2});
        OV_EXPECT_THROW(std::ignore = make_op(data, output_size_i32, kernel_size_i64),
                        ov::NodeValidationFailure,
                        HasSubstr(error_substring));
    }
}

TEST_F(TypePropCol2ImTest, batched_const_values) {
    PartialShape data_shape{3, 12, 81};
    auto data_symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::i64, data_shape);
    const auto output_size =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{16, 16});
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{2, 2};
    const auto pads_end = Shape{2, 2};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{2, 2}));
    EXPECT_EQ(op->get_pads_end(), (Shape{2, 2}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 3, 16, 16}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(data_symbols[0], nullptr, nullptr, nullptr));
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

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 32, 32}));
}

TEST_F(TypePropCol2ImTest, kernel_size_and_output_size_from_shapeof) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{12, 324});
    const auto output_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, Shape{32, 32}));
    const auto kernel_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, Shape{2, 2}));
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
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

TEST_F(TypePropCol2ImTest, dynamic_output_size) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{12, 324});
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, dynamic_kernel_size) {
    const auto data = std::make_shared<Parameter>(element::i64, Shape{12, 324});
    const auto output_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, Shape{32, 32}));
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 32, 32}));
}

TEST_F(TypePropCol2ImTest, dynamic_batch_size) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic(), 12, 324});
    const auto output_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, Shape{32, 32}));
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 3, 32, 32}));
}

TEST_F(TypePropCol2ImTest, dynamic_output_size_from_shape_of) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic(), 12, 324});
    const auto output_size = std::make_shared<ShapeOf>(
        std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, dynamic_kernel_size_from_shape_of) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic(), 12, 324});
    const auto output_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, Shape{32, 32}));
    const auto kernel_size = std::make_shared<ShapeOf>(
        std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
}

TEST_F(TypePropCol2ImTest, interval_data_shape) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape{{4, 5}, {12, 16}, {324, 623}});
    const auto output_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, Shape{32, 32}));
    const auto kernel_size = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int32_t>{2, 2});
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 5}, Dimension::dynamic(), 32, 32}));
}

TEST_F(TypePropCol2ImTest, dynamic_input_shapes) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST_F(TypePropCol2ImTest, static_batch) {
    PartialShape data_shape{5, Dimension::dynamic(), Dimension::dynamic()};
    auto data_symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::i64, data_shape);
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto kernel_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{5, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(data_symbols[0], nullptr, nullptr, nullptr));
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

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST_F(TypePropCol2ImTest, interval_inputs_from_shapeof) {
    PartialShape data_shape{{4, 5}, {12, 16}, {324, 623}};
    auto data_symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::i64, data_shape);
    const auto kernel_size =
        std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, PartialShape{{2, 32}, {2, 32}}));
    PartialShape output_size_shape{{5, 16}, {5, 16}};
    auto output_size_symbols = set_shape_symbols(output_size_shape);
    const auto output_size = std::make_shared<ShapeOf>(std::make_shared<Parameter>(element::i64, output_size_shape));
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{3, 3};
    const auto pads_end = Shape{3, 3};

    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), (Strides{2, 2}));
    EXPECT_EQ(op->get_dilations(), (Strides{2, 2}));
    EXPECT_EQ(op->get_pads_begin(), (Shape{3, 3}));
    EXPECT_EQ(op->get_pads_end(), (Shape{3, 3}));
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 5}, Dimension::dynamic(), {5, 16}, {5, 16}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(data_symbols[0], nullptr, output_size_symbols[0], output_size_symbols[1]));
}

}  // namespace test
}  // namespace ov
