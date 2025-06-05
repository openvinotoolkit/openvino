// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::test {
using testing::HasSubstr;

class TypePropSparseFillEmptyRowsUnpackedStringTest : public TypePropOpTest<op::v16::SparseFillEmptyRowsUnpackedString> {};

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, default_ctor_valid_inputs) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_element_type(3), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{10}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{3}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, with_const_inputs) {
    const auto begins = std::make_shared<op::v0::Constant>(element::i32, Shape{2, 2}, std::vector<int32_t>{0, 5, 10, 15});
    const auto ends = std::make_shared<op::v0::Constant>(element::i32, Shape{2, 2}, std::vector<int32_t>{5, 10, 10, 20});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{20});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_element_type(3), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{20}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{2}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, dynamic_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape::dynamic());
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape::dynamic());

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_element_type(3), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, partially_dynamic_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_element_type(1), element::i64);
    EXPECT_EQ(op->get_output_element_type(2), element::u8);
    EXPECT_EQ(op->get_output_element_type(3), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, known_row_count) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{5, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{5, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{5}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, known_symbols_length) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Constant>(element::u8, Shape{10}, std::vector<uint8_t>(10, 'A'));
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{10}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{3}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, known_default_value) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{30});
    const auto default_value = std::make_shared<op::v0::Constant>(element::u8, Shape{5}, std::vector<uint8_t>{'E', 'm', 'p', 't', 'y'});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{30}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{3}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, symbol_propagation) {
    PartialShape begins_shape{Dimension::dynamic(), 2};
    PartialShape ends_shape{Dimension::dynamic(), 2};
    PartialShape symbols_shape{Dimension::dynamic()};
    PartialShape default_value_shape{};
    
    auto begins_symbols = set_shape_symbols(begins_shape);
    auto ends_symbols = set_shape_symbols(ends_shape);
    auto symbols_symbols = set_shape_symbols(symbols_shape);
    
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, begins_shape);
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, ends_shape);
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, symbols_shape);
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, default_value_shape);

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{Dimension::dynamic()}));

    const auto& output0_symbols = get_shape_symbols(op->get_output_partial_shape(0));
    const auto& output1_symbols = get_shape_symbols(op->get_output_partial_shape(1));
    const auto& output2_symbols = get_shape_symbols(op->get_output_partial_shape(2));
    const auto& output3_symbols = get_shape_symbols(op->get_output_partial_shape(3));
    
    EXPECT_EQ(output0_symbols[0], begins_symbols[0]);
    EXPECT_EQ(output0_symbols[1], begins_symbols[1]);
    EXPECT_EQ(output1_symbols[0], ends_symbols[0]);
    EXPECT_EQ(output1_symbols[1], ends_symbols[1]);
    EXPECT_EQ(output2_symbols[0], symbols_symbols[0]);
    EXPECT_EQ(output3_symbols[0], begins_symbols[0]);
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, interval_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{3, 6}, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{3, 6}, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{{20, 30}});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();
    
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{3, 6}, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{3, 6}, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{{20, 30}}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{{3, 6}}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, empty_row_detection) {
    const auto begins = std::make_shared<op::v0::Constant>(element::i32, Shape{3, 2}, 
                        std::vector<int32_t>{0, 5, 10, 10, 15, 20});  // Row 1 has empty strings
    const auto ends = std::make_shared<op::v0::Constant>(element::i32, Shape{3, 2}, 
                      std::vector<int32_t>{5, 10, 10, 10, 20, 25});   // Same indices for row 1
    const auto symbols = std::make_shared<op::v0::Constant>(element::u8, Shape{25}, std::vector<uint8_t>(25, 'A'));
    const auto default_value = std::make_shared<op::v0::Constant>(element::u8, Shape{5}, 
                               std::vector<uint8_t>{'E', 'm', 'p', 't', 'y'});

    const auto op = make_op(begins, ends, symbols, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{30})); // Original 25 + 5 for "Empty"
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{3}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_begins_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The begins and ends inputs must have identical shapes"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_ends_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The begins and ends inputs must have identical shapes"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_symbols_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The symbols input must be a 1D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_begins_ends_element_types) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element types of the begins and ends inputs must match"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_merged_type) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 2});
    const auto ends = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 2});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element type of the begins and ends inputs must be i32 or i64"));
}

}  // namespace ov::test
