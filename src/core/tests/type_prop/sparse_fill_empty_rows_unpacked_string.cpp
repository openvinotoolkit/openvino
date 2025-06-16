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

class TypePropSparseFillEmptyRowsUnpackedStringTest
    : public TypePropOpTest<op::v16::SparseFillEmptyRowsUnpackedString> {};

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, default_ctor_valid_inputs) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::i32);
    EXPECT_EQ(op->get_output_element_type(3), element::u8);
    EXPECT_EQ(op->get_output_element_type(4), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{10}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, with_const_inputs) {
    const auto begins =
        std::make_shared<op::v0::Constant>(element::i32, Shape{6}, std::vector<int32_t>{0, 5, 10, 15, 20, 25});
    const auto ends =
        std::make_shared<op::v0::Constant>(element::i32, Shape{6}, std::vector<int32_t>{5, 10, 15, 20, 25, 30});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{30});
    const auto indices = std::make_shared<op::v0::Constant>(element::i32,
                                                            Shape{6, 2},
                                                            std::vector<int32_t>{0, 0, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{5, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::i32);
    EXPECT_EQ(op->get_output_element_type(3), element::u8);
    EXPECT_EQ(op->get_output_element_type(4), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{8, 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{35}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{5}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, dynamic_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape::dynamic());
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape::dynamic());

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_element_type(2), element::i32);
    EXPECT_EQ(op->get_output_element_type(3), element::u8);
    EXPECT_EQ(op->get_output_element_type(4), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(2), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(3), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(4), PartialShape{Dimension::dynamic()});
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, partially_dynamic_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_element_type(1), element::i64);
    EXPECT_EQ(op->get_output_element_type(2), element::i64);
    EXPECT_EQ(op->get_output_element_type(3), element::u8);
    EXPECT_EQ(op->get_output_element_type(4), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, known_row_count) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5, 2});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{10, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{5}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{5}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{5, 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{10}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, known_symbols_length) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Constant>(element::u8, Shape{10}, std::vector<uint8_t>(10, 'A'));
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{10}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, known_default_value) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{30});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value =
        std::make_shared<op::v0::Constant>(element::u8, Shape{5}, std::vector<uint8_t>{'E', 'm', 'p', 't', 'y'});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{30}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, symbol_propagation) {
    PartialShape begins_shape{Dimension::dynamic()};
    PartialShape ends_shape{Dimension::dynamic()};
    PartialShape symbols_shape{Dimension::dynamic()};
    PartialShape indices_shape{Dimension::dynamic(), 2};
    PartialShape dense_shape_shape{Dimension::dynamic()};

    auto begins_symbols = set_shape_symbols(begins_shape);
    auto ends_symbols = set_shape_symbols(ends_shape);
    auto symbols_symbols = set_shape_symbols(symbols_shape);
    auto indices_symbols = set_shape_symbols(indices_shape);

    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, begins_shape);
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, ends_shape);
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, symbols_shape);
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, indices_shape);
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, dense_shape_shape);
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{Dimension::dynamic()}));

    const auto& out_begins_symbols = get_shape_symbols(op->get_output_partial_shape(0));
    const auto& out_ends_symbols = get_shape_symbols(op->get_output_partial_shape(1));
    const auto& out_indices_symbols = get_shape_symbols(op->get_output_partial_shape(2));
    const auto& out_symbols_symbols = get_shape_symbols(op->get_output_partial_shape(3));
    const auto& out_empty_row_indicator_symbols = get_shape_symbols(op->get_output_partial_shape(4));

    EXPECT_EQ(out_begins_symbols[0], begins_symbols[0]);
    EXPECT_EQ(out_ends_symbols[0], ends_symbols[0]);
    EXPECT_EQ(out_indices_symbols[0], indices_symbols[0]);
    EXPECT_EQ(out_indices_symbols[1], indices_symbols[1]);
    EXPECT_EQ(out_symbols_symbols[0], symbols_symbols[0]);
    EXPECT_EQ(out_empty_row_indicator_symbols[0], nullptr);
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, interval_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{3, 6}});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{3, 6}});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{{20, 30}});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{3, 6}, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{Dimension::dynamic()});

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{3, 6}}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{3, 6}}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{{3, 6}, 2}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{{20, 30}}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_symbols_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5, 2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The symbols input must be a 1D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_begins_ends_element_types) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element types of the begins and ends inputs must match"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_merged_type) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element type of the index inputs must be i32 or i64"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_indices_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The indices input must be a 2D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_dense_shape_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The dense_shape input must be a 1D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, invalid_default_value_rank) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{2, 2});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The default_value input must be a 1D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsUnpackedStringTest, incompatible_begins_indices_shapes) {
    const auto begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto ends = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto symbols = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{10});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{5});

    OV_EXPECT_THROW(std::ignore = make_op(begins, ends, symbols, indices, dense_shape, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The begins and indices inputs must have the same shape"));
}
}  // namespace ov::test
