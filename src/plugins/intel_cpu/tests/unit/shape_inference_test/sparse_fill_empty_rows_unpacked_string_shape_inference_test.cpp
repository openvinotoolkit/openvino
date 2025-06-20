// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"
#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"
#include "openvino/op/constant.hpp"

using namespace ov::intel_cpu;
using ov::op::v0::Constant, ov::op::v0::Parameter;
using testing::HasSubstr;

struct SparseFillEmptyRowsUnpackedStringTestParams {
    ov::Shape symbols_shape;
    ov::Shape default_value_shape;
    std::vector<int32_t> begins_val;
    std::vector<int32_t> ends_val;
    std::vector<int32_t> indices_val;
    std::vector<int32_t> dense_shape_val;
    ov::Shape expected_output_begins_shape;
    ov::Shape expected_output_indices_shape;
    ov::Shape expected_output_symbols_shape;
};

class SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTest:
    public OpStaticShapeInferenceTest<ov::op::v16::SparseFillEmptyRowsUnpackedString> {};

class SparseFillEmptyRowsUnpackedStringStaticTestSuite:
    public SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTest,
    public ::testing::WithParamInterface<SparseFillEmptyRowsUnpackedStringTestParams> {};

TEST_F(SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTest, input_from_tensor_accessor) {
    const auto begins = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto ends = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto symbols = std::make_shared<Parameter>(ov::element::u8, ov::PartialShape::dynamic());
    const auto indices = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto dense_shape = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto default_value = std::make_shared<Parameter>(ov::element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);

    int32_t begins_val[] = {0, 5, 10, 15, 20, 25, 30};
    int32_t ends_val[] = {5, 10, 15, 20, 25, 30, 35};
    uint8_t symbols_val[35];
    std::fill_n(symbols_val, 35, 'A');
    int32_t indices_val[] = {0, 0, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0};
    int32_t dense_shape_val[] = {5, 2};
    uint8_t default_value_val[] = {'E', 'm', 'p', 't', 'y'};

    auto const_inputs = std::unordered_map<size_t, ov::Tensor>{
        {0, {ov::element::i32, ov::Shape{7}, begins_val}},
        {1, {ov::element::i32, ov::Shape{7}, ends_val}},
        {2, {ov::element::u8, ov::Shape{35}, symbols_val}},
        {3, {ov::element::i32, ov::Shape{7, 2}, indices_val}},
        {4, {ov::element::i32, ov::Shape{2}, dense_shape_val}},
        {5, {ov::element::u8, ov::Shape{5}, default_value_val}}
    };

    const auto input_shapes = StaticShapeVector{{7}, {7}, {35}, {7, 2}, {2}, {5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor(const_inputs));

    EXPECT_EQ(output_shapes.size(), 5);
    EXPECT_EQ(output_shapes[0], StaticShape({8}));      // output_begins
    EXPECT_EQ(output_shapes[1], StaticShape({8}));      // output_ends
    EXPECT_EQ(output_shapes[2], StaticShape({8, 2}));   // output_indices
    EXPECT_EQ(output_shapes[3], StaticShape({40}));     // output_symbols (35 + 5 for "Empty")
    EXPECT_EQ(output_shapes[4], StaticShape({5}));      // empty_row_indicator
}

TEST_F(SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTest, static_shapes) {
    const auto begins = std::make_shared<Parameter>(ov::element::i32, ov::Shape{6});
    const auto ends = std::make_shared<Parameter>(ov::element::i32, ov::Shape{6});
    const auto symbols = std::make_shared<Parameter>(ov::element::u8, ov::Shape{30});
    const auto indices = std::make_shared<Parameter>(ov::element::i32, ov::Shape{6, 2});
    const auto dense_shape = std::make_shared<Parameter>(ov::element::i32, ov::Shape{2});
    const auto default_value = std::make_shared<Parameter>(ov::element::u8, ov::Shape{5});
    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);

    int32_t begins_val[] = {0, 5, 10, 15, 20, 25};
    int32_t ends_val[] = {5, 10, 15, 20, 25, 30};
    uint8_t symbols_val[30];
    std::fill_n(symbols_val, 30, 'A');
    int32_t indices_val[] = {0, 0, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1};
    int32_t dense_shape_val[] = {5, 2};
    uint8_t default_value_val[] = {'E', 'm', 'p', 't', 'y'};

    auto const_inputs = std::unordered_map<size_t, ov::Tensor>{
        {0, {ov::element::i32, ov::Shape{6}, begins_val}},
        {1, {ov::element::i32, ov::Shape{6}, ends_val}},
        {2, {ov::element::u8, ov::Shape{30}, symbols_val}},
        {3, {ov::element::i32, ov::Shape{6, 2}, indices_val}},
        {4, {ov::element::i32, ov::Shape{2}, dense_shape_val}},
        {5, {ov::element::u8, ov::Shape{5}, default_value_val}}
    };

    const auto input_shapes = StaticShapeVector{{6}, {6}, {30}, {6, 2}, {2}, {5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor(const_inputs));

    EXPECT_EQ(output_shapes.size(), 5);
    EXPECT_EQ(output_shapes[0], StaticShape({8}));      // output_begins
    EXPECT_EQ(output_shapes[1], StaticShape({8}));      // output_ends
    EXPECT_EQ(output_shapes[2], StaticShape({8, 2}));   // output_indices
    EXPECT_EQ(output_shapes[3], StaticShape({35}));     // output_symbols (30 + 5 for "Empty")
    EXPECT_EQ(output_shapes[4], StaticShape({5}));      // empty_row_indicator
}

TEST_P(SparseFillEmptyRowsUnpackedStringStaticTestSuite, sparse_fill_empty_rows_unpacked_string_static_shape_inference) {
    const auto& [symbols_shape, default_value_shape,
                begins_val, ends_val, indices_val, dense_shape_val,
                expected_output_begins_shape, expected_output_indices_shape,
                expected_output_symbols_shape] = GetParam();

    const ov::Shape begins_shape{begins_val.size()};
    const ov::Shape indices_shape{indices_val.size() / 2, 2};

    const auto begins = std::make_shared<Constant>(ov::element::i32, begins_shape, begins_val);
    const auto ends = std::make_shared<Constant>(ov::element::i32, begins_shape, ends_val);
    const auto symbols = std::make_shared<Parameter>(ov::element::u8, symbols_shape);
    const auto indices = std::make_shared<Constant>(ov::element::i32, indices_shape, indices_val);
    const auto dense_shape = std::make_shared<Constant>(ov::element::i32, ov::Shape{2}, dense_shape_val);

    std::vector<uint8_t> default_value_data(default_value_shape.size() > 0 ? default_value_shape[0] : 0);
    for (size_t i = 0; i < default_value_data.size(); ++i) {
        static const char* empty_str = "Empty";
        default_value_data[i] = i < 5 ? static_cast<uint8_t>(empty_str[i]) : 0;
    }
    const auto default_value = std::make_shared<Constant>(ov::element::u8, default_value_shape, default_value_data);

    const auto op = make_op(begins, ends, symbols, indices, dense_shape, default_value);

    const auto input_shapes = StaticShapeVector{begins_shape, begins_shape, symbols_shape,
                                              indices_shape, ov::Shape{2}, default_value_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 5);
    EXPECT_EQ(output_shapes[0], StaticShape(expected_output_begins_shape));                 // output_begins
    EXPECT_EQ(output_shapes[1], StaticShape(expected_output_begins_shape));                 // output_ends
    EXPECT_EQ(output_shapes[2], StaticShape(expected_output_indices_shape));                // output_indices
    EXPECT_EQ(output_shapes[3], StaticShape(expected_output_symbols_shape));                // output_symbols
    EXPECT_EQ(output_shapes[4], StaticShape({static_cast<size_t>(dense_shape_val[0])}));    // empty_row_indicator
}

INSTANTIATE_TEST_SUITE_P(SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTests,
                         SparseFillEmptyRowsUnpackedStringStaticTestSuite,
                         ::testing::Values(
    // No empty rows
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{30},                                              // symbols_shape
        ov::Shape{5},                                               // default_value_shape
        std::vector<int32_t>{0, 5, 10, 15, 20, 25},                 // begins_val
        std::vector<int32_t>{5, 10, 15, 20, 25, 30},                // ends_val
        std::vector<int32_t>{0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1},   // indices_val
        std::vector<int32_t>{3, 2},                                 // dense_shape_val
        ov::Shape{6},                                               // expected_output_begins_shape
        ov::Shape{6, 2},                                            // expected_output_indices_shape
        ov::Shape{30}                                               // expected_output_symbols_shape
    },
    // Row 1 is empty
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{30},                                              // symbols_shape
        ov::Shape{5},                                               // default_value_shape
        std::vector<int32_t>{0, 5, 10, 15, 20, 25},                 // begins_val
        std::vector<int32_t>{5, 10, 15, 20, 25, 30},                // ends_val
        std::vector<int32_t>{0, 0, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1},   // indices_val (no row 1)
        std::vector<int32_t>{4, 2},                                 // dense_shape_val
        ov::Shape{7},                                               // expected_output_begins_shape
        ov::Shape{7, 2},                                            // expected_output_indices_shape
        ov::Shape{35}                                               // expected_output_symbols_shape (30 + 5 for "Empty")
    },
    // Multiple empty rows (1 and 3)
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{40},                                              // symbols_shape
        ov::Shape{5},                                               // default_value_shape
        std::vector<int32_t>{0, 5, 10, 15, 20, 25},                 // begins_val
        std::vector<int32_t>{5, 10, 15, 20, 25, 30},                // ends_val
        std::vector<int32_t>{0, 0, 0, 1, 2, 0, 2, 1, 4, 0, 4, 1},   // indices_val (no rows 1 and 3)
        std::vector<int32_t>{5, 2},                                 // dense_shape_val
        ov::Shape{8},                                               // expected_output_begins_shape
        ov::Shape{8, 2},                                            // expected_output_indices_shape
        ov::Shape{45}                                               // expected_output_symbols_shape (40 + 5 for "Empty")
    },
    // Empty default_value
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{25},                                      // symbols_shape
        ov::Shape{0},                                       // default_value_shape (empty string)
        std::vector<int32_t>{0, 5, 10, 15, 20},             // begins_val
        std::vector<int32_t>{5, 10, 15, 20, 25},            // ends_val
        std::vector<int32_t>{0, 0, 0, 1, 2, 0, 2, 1, 3, 0}, // indices_val (no rows 1 and 4)
        std::vector<int32_t>{5, 2},                         // dense_shape_val
        ov::Shape{7},                                       // expected_output_begins_shape
        ov::Shape{7, 2},                                    // expected_output_indices_shape
        ov::Shape{25}                                       // expected_output_symbols_shape (unchanged)
    },
    // All rows are empty
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{0},                                       // symbols_shape
        ov::Shape{5},                                       // default_value_shape
        std::vector<int32_t>{},                             // begins_val (empty, no entries)
        std::vector<int32_t>{},                             // ends_val (empty, no entries)
        std::vector<int32_t>{},                             // indices_val (empty, no entries)
        std::vector<int32_t>{3, 2},                         // dense_shape_val (3 rows, all empty)
        ov::Shape{3},                                       // expected_output_begins_shape (one entry per row)
        ov::Shape{3, 2},                                    // expected_output_indices_shape (one entry per row)
        ov::Shape{5}                                        // expected_output_symbols_shape (just the default value)
    }
));
