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
    ov::Shape indices_shape;
    ov::Shape symbols_shape;
    ov::Shape default_value_shape;
    std::vector<int32_t> begins_val;
    std::vector<int32_t> ends_val;
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
    const auto default_value = std::make_shared<Parameter>(ov::element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins, ends, symbols, default_value);

    int32_t begins_val[] = {0, 5, 10, 10, 20, 25, 30, 35};  // Row 1 has empty strings
    int32_t ends_val[] = {5, 10, 10, 10, 25, 30, 35, 40};
    uint8_t symbols_val[40];
    std::fill_n(symbols_val, 40, 'A');
    uint8_t default_value_val[] = {'E', 'm', 'p', 't', 'y'};

    auto const_inputs = std::unordered_map<size_t, ov::Tensor>{
        {0, {ov::element::i32, ov::Shape{4, 2}, begins_val}},
        {1, {ov::element::i32, ov::Shape{4, 2}, ends_val}},
        {2, {ov::element::u8, ov::Shape{40}, symbols_val}},
        {3, {ov::element::u8, ov::Shape{5}, default_value_val}}
    };

    const auto input_shapes = StaticShapeVector{{4, 2}, {4, 2}, {40}, {5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor(const_inputs));

    EXPECT_EQ(output_shapes.size(), 4);
    EXPECT_EQ(output_shapes[0], StaticShape({4, 2}));  // output_begins
    EXPECT_EQ(output_shapes[1], StaticShape({4, 2}));  // output_ends
    EXPECT_EQ(output_shapes[2], StaticShape({45}));    // output_symbols (40 + 5 for "Empty")
    EXPECT_EQ(output_shapes[3], StaticShape({4}));     // empty_row_indicator
}

TEST_F(SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTest, static_shapes) {
    const auto begins = std::make_shared<Parameter>(ov::element::i32, ov::Shape{3, 2});
    const auto ends = std::make_shared<Parameter>(ov::element::i32, ov::Shape{3, 2});
    const auto symbols = std::make_shared<Parameter>(ov::element::u8, ov::Shape{30});
    const auto default_value = std::make_shared<Parameter>(ov::element::u8, ov::Shape{5});
    const auto op = make_op(begins, ends, symbols, default_value);

    int32_t begins_val[] = {0, 5, 10, 10, 20, 25};  // Row 1 has empty strings
    int32_t ends_val[] = {5, 10, 10, 10, 25, 30};
    uint8_t symbols_val[30];
    std::fill_n(symbols_val, 30, 'A');
    uint8_t default_value_val[] = {'E', 'm', 'p', 't', 'y'};

    auto const_inputs = std::unordered_map<size_t, ov::Tensor>{
        {0, {ov::element::i32, ov::Shape{3, 2}, begins_val}},
        {1, {ov::element::i32, ov::Shape{3, 2}, ends_val}},
        {2, {ov::element::u8, ov::Shape{30}, symbols_val}},
        {3, {ov::element::u8, ov::Shape{5}, default_value_val}}
    };

    const auto input_shapes = StaticShapeVector{{3, 2}, {3, 2}, {30}, {5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor(const_inputs));

    EXPECT_EQ(output_shapes.size(), 4);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 2}));  // output_begins
    EXPECT_EQ(output_shapes[1], StaticShape({3, 2}));  // output_ends
    EXPECT_EQ(output_shapes[2], StaticShape({35}));    // output_symbols (30 + 5 for "Empty")
    EXPECT_EQ(output_shapes[3], StaticShape({3}));     // empty_row_indicator
}

TEST_P(SparseFillEmptyRowsUnpackedStringStaticTestSuite, sparse_fill_empty_rows_unpacked_string_static_shape_inference) {
    const auto& [indices_shape, symbols_shape, default_value_shape,
                begins_val, ends_val,
                expected_output_indices_shape,
                expected_output_symbols_shape] = GetParam();

    const auto begins = std::make_shared<Constant>(ov::element::i32, indices_shape, begins_val);
    const auto ends = std::make_shared<Constant>(ov::element::i32, indices_shape, ends_val);
    const auto symbols = std::make_shared<Parameter>(ov::element::u8, symbols_shape);
    std::vector<uint8_t> default_value_data(default_value_shape.size() > 0 ? default_value_shape[0] : 0);
    for (size_t i = 0; i < default_value_data.size(); ++i) {
        static const char* empty_str = "Empty";
        default_value_data[i] = i < 5 ? static_cast<uint8_t>(empty_str[i]) : 0;
    }
    const auto default_value = std::make_shared<Constant>(ov::element::u8, default_value_shape, default_value_data);

    const auto op = make_op(begins, ends, symbols, default_value);

    const auto input_shapes = StaticShapeVector{indices_shape, indices_shape, symbols_shape, default_value_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 4);
    EXPECT_EQ(output_shapes[0], StaticShape(expected_output_indices_shape));
    EXPECT_EQ(output_shapes[1], StaticShape(expected_output_indices_shape));
    EXPECT_EQ(output_shapes[2], StaticShape(expected_output_symbols_shape));
    EXPECT_EQ(output_shapes[3], StaticShape({indices_shape[0]}));
}

INSTANTIATE_TEST_SUITE_P(SparseFillEmptyRowsUnpackedStringStaticShapeInferenceTests,
                         SparseFillEmptyRowsUnpackedStringStaticTestSuite,
                         ::testing::Values(
    // No empty rows
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{3, 2},                                // indices_shape (for both begins and ends)
        ov::Shape{30},                                  // symbols_shape
        ov::Shape{5},                                   // default_value_shape
        std::vector<int32_t>{0, 5, 10, 15, 20, 25},     // begins_val
        std::vector<int32_t>{5, 10, 15, 20, 25, 30},    // ends_val
        ov::Shape{3, 2},                                // expected_output_indices_shape
        ov::Shape{30}                                   // expected_output_symbols_shape
    },
    // Row 1 is empty
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{3, 2},                                    // indices_shape
        ov::Shape{30},                                      // symbols_shape
        ov::Shape{5},                                       // default_value_shape
        std::vector<int32_t>{0, 5, 10, 10, 15, 20},         // begins_val (row 1 has empty strings)
        std::vector<int32_t>{5, 10, 10, 10, 20, 25},        // ends_val (row 1 has empty strings)
        ov::Shape{3, 2},                                    // expected_output_indices_shape
        ov::Shape{35}                                       // expected_output_symbols_shape (30 + 5 for "Empty")
    },
    // Multiple empty rows (1 and 3)
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{4, 2},                                        // indices_shape
        ov::Shape{40},                                          // symbols_shape
        ov::Shape{5},                                           // default_value_shape
        std::vector<int32_t>{0, 5, 10, 10, 15, 20, 25, 25},     // begins_val (rows 1 and 3 have empty strings)
        std::vector<int32_t>{5, 10, 10, 10, 20, 25, 25, 25},    // ends_val (rows 1 and 3 have empty strings)
        ov::Shape{4, 2},                                        // expected_output_indices_shape
        ov::Shape{45}                                           // expected_output_symbols_shape (40 + 5 for "Empty")
    },
    // All rows empty
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{3, 2},                            // indices_shape
        ov::Shape{30},                              // symbols_shape
        ov::Shape{5},                               // default_value_shape
        std::vector<int32_t>{0, 0, 10, 10, 20, 20}, // begins_val
        std::vector<int32_t>{0, 0, 10, 10, 20, 20}, // ends_val
        ov::Shape{3, 2},                            // expected_output_indices_shape
        ov::Shape{35}                               // expected_output_symbols_shape (30 + 3*5 for three "Empty" strings)
    },
    // Mixed string lengths with one empty row
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{3, 2},                             // indices_shape
        ov::Shape{25},                               // symbols_shape
        ov::Shape{5},                                // default_value_shape
        std::vector<int32_t>{0, 5, 10, 10, 15, 20},  // begins_val
        std::vector<int32_t>{5, 10, 10, 10, 20, 25}, // ends_val
        ov::Shape{3, 2},                             // expected_output_indices_shape
        ov::Shape{30}                                // expected_output_symbols_shape (25 + 5 for "Empty")
    },
    // Empty default_value
    SparseFillEmptyRowsUnpackedStringTestParams{
        ov::Shape{3, 2},                             // indices_shape
        ov::Shape{25},                               // symbols_shape
        ov::Shape{0},                                // default_value_shape (empty string)
        std::vector<int32_t>{0, 5, 10, 10, 15, 20},  // begins_val
        std::vector<int32_t>{5, 10, 10, 10, 20, 25}, // ends_val
        ov::Shape{3, 2},                             // expected_output_indices_shape
        ov::Shape{25}                                // expected_output_symbols_shape (unchanged since default_value is empty)
    }
));
