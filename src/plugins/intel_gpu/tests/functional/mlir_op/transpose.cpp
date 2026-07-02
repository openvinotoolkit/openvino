// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// Params: input shape, order, precision
using TransposeParams = std::tuple<ov::Shape, std::vector<int64_t>, ov::element::Type>;

class TransposeTest : public testing::WithParamInterface<TransposeParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeParams>& obj) {
        const auto& [shape, order, precision] = obj.param;
        std::ostringstream result;
        result << "Input=" << ov::test::utils::vec2str(shape) << "_";
        result << "Order=" << ov::test::utils::vec2str(order) << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [shape, order, precision] = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        auto order_const = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
        auto op = std::make_shared<ov::op::v1::Transpose>(input, order_const);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
};

TEST_P(TransposeTest, Inference) {
    run();
}

const auto transpose_params = ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 24, 64}, ov::Shape{1, 128, 24, 64}),
                                                 ::testing::Values(std::vector<int64_t>{0, 2, 1, 3}),
                                                 ::testing::Values(ov::element::f16));
INSTANTIATE_TEST_SUITE_P(mlir_Transpose, TransposeTest, transpose_params, TransposeTest::getTestCaseName);

// Params: input shape, output shape, order, precision
using ReshapeAndTransposeParams = std::tuple<ov::Shape, ov::Shape, std::vector<int64_t>, ov::element::Type>;

class ReshapeAndTransposeTest : public testing::WithParamInterface<ReshapeAndTransposeParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeAndTransposeParams>& obj) {
        const auto& [input_shape, output_shape, order, precision] = obj.param;
        std::ostringstream result;
        result << "Input=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "Output=" << ov::test::utils::vec2str(output_shape) << "_";
        result << "Order=" << ov::test::utils::vec2str(order) << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, output_shape, order, precision] = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        std::vector<int64_t> shape_data(output_shape.begin(), output_shape.end());
        auto shape_const = ov::op::v0::Constant::create(ov::element::i64, {output_shape.size()}, shape_data);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape_const, false);
        auto order_const = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
        auto op = std::make_shared<ov::op::v1::Transpose>(reshape, order_const);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
};

TEST_P(ReshapeAndTransposeTest, Inference) {
    run();
}

const auto reshape_and_transpose_params =
    ::testing::Values(ReshapeAndTransposeParams{ov::Shape{1, 1024, 1536}, ov::Shape{1, 1024, 24, 64}, std::vector<int64_t>{0, 2, 1, 3}, ov::element::f16},
                      ReshapeAndTransposeParams{ov::Shape{1, 128, 1536}, ov::Shape{1, 128, 24, 64}, std::vector<int64_t>{0, 2, 1, 3}, ov::element::f16});
INSTANTIATE_TEST_SUITE_P(mlir_ReshapeAndTranspose, ReshapeAndTransposeTest, reshape_and_transpose_params, ReshapeAndTransposeTest::getTestCaseName);

}  // namespace
