// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bincount_shape_inference.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using TensorMap = std::unordered_map<size_t, ov::Tensor>;

class BincountV17StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v17::Bincount> {
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(BincountV17StaticShapeInferenceTest, unweighted_with_constant_data) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    op = make_op(data, 0);

    input_shapes = StaticShapeVector{{5}};
    // data = [0, 1, 2, 3, 0] -> max is 3 -> output size = 4
    const std::vector<int32_t> vals = {0, 1, 2, 3, 0};
    auto data_tensor = ov::Tensor(element::i32, ov::Shape{5});
    std::memcpy(data_tensor.data(), vals.data(), vals.size() * sizeof(int32_t));
    TensorMap constant_data;
    constant_data[0] = data_tensor;

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({4}));
}

TEST_F(BincountV17StaticShapeInferenceTest, minlength_larger_than_max_value) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    op = make_op(data, 10);  // minlength = 10

    input_shapes = StaticShapeVector{{3}};
    // data = [0, 1, 2] -> max is 2 -> output size = max(3, 10) = 10
    const std::vector<int32_t> vals = {0, 1, 2};
    auto data_tensor = ov::Tensor(element::i32, ov::Shape{3});
    std::memcpy(data_tensor.data(), vals.data(), vals.size() * sizeof(int32_t));
    TensorMap constant_data;
    constant_data[0] = data_tensor;

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10}));
}

TEST_F(BincountV17StaticShapeInferenceTest, empty_data_with_zero_minlength) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    op = make_op(data, 0);

    input_shapes = StaticShapeVector{{0}};
    auto data_tensor = ov::Tensor(element::i32, ov::Shape{0});
    TensorMap constant_data;
    constant_data[0] = data_tensor;

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({0}));
}

TEST_F(BincountV17StaticShapeInferenceTest, i64_data) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    op = make_op(data, 0);

    input_shapes = StaticShapeVector{{4}};
    // data = [0, 5, 2, 5] -> max is 5 -> output size = 6
    const std::vector<int64_t> vals = {0, 5, 2, 5};
    auto data_tensor = ov::Tensor(element::i64, ov::Shape{4});
    std::memcpy(data_tensor.data(), vals.data(), vals.size() * sizeof(int64_t));
    TensorMap constant_data;
    constant_data[0] = data_tensor;

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({6}));
}

TEST_F(BincountV17StaticShapeInferenceTest, weighted_with_constant_data) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto weights = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    op = make_op(data, weights, 0);

    input_shapes = StaticShapeVector{{3}, {3}};
    // data = [0, 1, 2] -> max is 2 -> output size = 3
    const std::vector<int32_t> vals = {0, 1, 2};
    auto data_tensor = ov::Tensor(element::i32, ov::Shape{3});
    std::memcpy(data_tensor.data(), vals.data(), vals.size() * sizeof(int32_t));
    TensorMap constant_data;
    constant_data[0] = data_tensor;

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3}));
}
