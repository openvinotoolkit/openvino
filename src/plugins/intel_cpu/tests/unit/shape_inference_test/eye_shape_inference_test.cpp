// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gmock/gmock.h"
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
    const auto rows = Constant::create(element::i32, Shape{1}, {5});
    const auto cols = Constant::create(element::i32, Shape{1}, {4});
    const auto diag = Constant::create(element::i32, Shape{}, {1});
    const auto batch = Constant::create(element::i32, Shape{1}, {2});

    const auto op = make_op(rows, cols, diag, batch, element::f64);

    input_shapes = ShapeVector{rows->get_shape(), cols->get_shape(), diag->get_shape(), batch->get_shape()};
    shape_inference(op.get(), input_shapes, output_shapes, {});

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
    const auto const_data =
        std::map<size_t, HostTensorPtr>{{0, std::make_shared<HostTensor>(element::i32, Shape{}, &rows)},
                                        {1, std::make_shared<HostTensor>(element::i32, Shape{1}, &cols)},
                                        {3, std::make_shared<HostTensor>(element::i32, Shape{3}, batch.data())}};

    input_shapes = ShapeVector{{}, {1}, {1}, {3}};
    shape_inference(op.get(), input_shapes, output_shapes, const_data);

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
        std::map<size_t, HostTensorPtr>{{0, std::make_shared<HostTensor>(element::i64, Shape{}, &rows)},
                                        {1, std::make_shared<HostTensor>(element::i64, Shape{1}, &cols)},
                                        {3, std::make_shared<HostTensor>(element::i32, Shape{3}, batch.data())}};

    input_shapes = ShapeVector{{}, {1}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("'num_rows' must be non-negative value. Got: "));
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
        std::map<size_t, HostTensorPtr>{{0, std::make_shared<HostTensor>(element::i64, Shape{}, &rows)},
                                        {1, std::make_shared<HostTensor>(element::i64, Shape{1}, &cols)},
                                        {3, std::make_shared<HostTensor>(element::i32, Shape{3}, batch.data())}};

    input_shapes = ShapeVector{{}, {1}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("'num_columns' must be non-negative value. Got: "));
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
    const auto const_data = std::map<size_t, HostTensorPtr>{
        {0, std::make_shared<HostTensor>(element::i64, Shape{rows.size()}, rows.data())},
        {1, std::make_shared<HostTensor>(element::i64, Shape{1}, &cols)},
        {3, std::make_shared<HostTensor>(element::i32, Shape{batch.size()}, batch.data())}};

    input_shapes = ShapeVector{{}, {1}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
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
    const auto const_data = std::map<size_t, HostTensorPtr>{
        {0, std::make_shared<HostTensor>(element::i64, Shape{}, &rows)},
        {1, std::make_shared<HostTensor>(element::i64, Shape{cols.size()}, cols.data())},
        {3, std::make_shared<HostTensor>(element::i32, Shape{batch.size()}, batch.data())}};

    input_shapes = ShapeVector{{1}, {}, {1}, {3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
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
    const auto const_data = std::map<size_t, HostTensorPtr>{
        {0, std::make_shared<HostTensor>(element::i64, Shape{}, &rows)},
        {1, std::make_shared<HostTensor>(element::i64, Shape{}, &cols)},
        {3, std::make_shared<HostTensor>(element::i32, Shape{batch.size()}, batch.data())}};

    input_shapes = ShapeVector{{}, {}, {}, {2}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Check 'batch_shape[0].get_length() == output_shape.rank().get_length()'"));
}
