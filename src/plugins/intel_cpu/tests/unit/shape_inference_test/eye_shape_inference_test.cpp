// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include <array>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class EyeV9StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v9::Eye> {
public:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(EyeV9StaticShapeInferenceTest, parameters_as_constant) {
    const auto rows = Constant::create(element::i32, ov::Shape{1}, {5});
    const auto cols = Constant::create(element::i32, ov::Shape{1}, {4});
    const auto diag = Constant::create(element::i32, ov::Shape{}, {1});
    const auto batch = Constant::create(element::i32, ov::Shape{1}, {2});

    const auto op = make_op(rows, cols, diag, batch, element::f64);

    input_shapes = StaticShapeVector{rows->get_shape(), cols->get_shape(), diag->get_shape(), batch->get_shape()};
    const auto output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 5, 4}));
}

TEST_F(EyeV9StaticShapeInferenceTest, parameters_in_const_data_map) {
    const auto rows_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto cols_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto diag_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto batch_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(rows_node, cols_node, diag_node, batch_node, element::f64);

    int32_t rows = 3, cols = 8;
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &rows}},
                                                                   {1, {element::i32, ov::Shape{1}, &cols}},
                                                                   {3, {element::i32, ov::Shape{3}, batch.data()}}};

    input_shapes = StaticShapeVector{{}, {1}, {1}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 4, 1, 3, 8}));
}

TEST_F(EyeV9StaticShapeInferenceTest, assert_on_negative_rows) {
    const auto rows_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto cols_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto diag_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto batch_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(rows_node, cols_node, diag_node, batch_node, element::i16);

    int64_t rows = -3, cols = 8;
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto const_data =
        std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &rows}},
                                               {1, {element::i32, ov::Shape{1}, &cols}},
                                               {3, {element::i32, ov::Shape{batch.size()}, batch.data()}}};

    input_shapes = StaticShapeVector{{}, {1}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    AssertFailure,
                    HasSubstr("Value -3 not in range [0:"));
}

TEST_F(EyeV9StaticShapeInferenceTest, assert_on_negative_columns) {
    const auto rows_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto cols_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto diag_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto batch_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(rows_node, cols_node, diag_node, batch_node, element::i16);

    int64_t rows = 3, cols = -8;
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto const_data =
        std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &rows}},
                                               {1, {element::i32, ov::Shape{1}, &cols}},
                                               {3, {element::i32, ov::Shape{batch.size()}, batch.data()}}};

    input_shapes = StaticShapeVector{{}, {1}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    AssertFailure,
                    HasSubstr("Value -8 not in range [0:"));
}

TEST_F(EyeV9StaticShapeInferenceTest, assert_on_rows_not_1D) {
    const auto rows_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto cols_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto diag_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto batch_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(rows_node, cols_node, diag_node, batch_node, element::i16);

    int64_t cols = 8;
    auto rows = std::array<int64_t, 2>{2, 1};
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto const_data =
        std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{rows.size()}, &rows}},
                                               {1, {element::i32, ov::Shape{1}, &cols}},
                                               {3, {element::i32, ov::Shape{batch.size()}, batch.data()}}};

    input_shapes = StaticShapeVector{{}, {1}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("'num_rows' value must be a scalar or 1D tensor. Got:"));
}

TEST_F(EyeV9StaticShapeInferenceTest, assert_on_columns_not_1D) {
    const auto rows_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto cols_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto diag_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto batch_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(rows_node, cols_node, diag_node, batch_node, element::i16);

    int64_t rows = 8;
    auto cols = std::array<int64_t, 2>{2, 1};
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto const_data =
        std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &rows}},
                                               {1, {element::i32, ov::Shape{cols.size()}, &cols}},
                                               {3, {element::i32, ov::Shape{batch.size()}, batch.data()}}};

    input_shapes = StaticShapeVector{{1}, {}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("'num_columns' value must be a scalar or 1D tensor. Got:"));
}

TEST_F(EyeV9StaticShapeInferenceTest, assert_on_batch_shape_not_match_shape_in_const_data) {
    const auto rows_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto cols_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto diag_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto batch_node = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(rows_node, cols_node, diag_node, batch_node, element::i16);

    int64_t rows = 8, cols = 5;
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto const_data =
        std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &rows}},
                                               {1, {element::i32, ov::Shape{}, &cols}},
                                               {3, {element::i32, ov::Shape{batch.size()}, batch.data()}}};

    input_shapes = StaticShapeVector{{}, {}, {}, {2}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Check 'static_cast<int64_t>(batch_shape[0].get_length()) == "
                              "static_cast<int64_t>(batch_as_shape->rank().get_length())'"));
}
