// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"
#include "openvino/op/transpose.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// Params: precision
using ConcatParams = ov::element::Type;

class ConcatTest : public testing::WithParamInterface<ConcatParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParams>& obj) {
        std::ostringstream result;
        result << "precision=" << obj.param;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto precision = GetParam();

        auto input0 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 24, 1024, 64});
        auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 24, 128, 64});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{input0, input1}, 2);

        auto result = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
};

TEST_P(ConcatTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(mlir_Concat, ConcatTest, ::testing::Values(ov::element::f16), ConcatTest::getTestCaseName);

// Params: precision
using TransposeConcatParams = ov::element::Type;

class TransposeConcatTest : public testing::WithParamInterface<TransposeConcatParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeConcatParams>& obj) {
        std::ostringstream result;
        result << "precision=" << obj.param;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto precision = GetParam();
        const std::vector<int64_t> order{0, 2, 1, 3};
        auto order_const = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);

        auto input0 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 1024, 24, 64});
        auto transpose0 = std::make_shared<ov::op::v1::Transpose>(input0, order_const);

        auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 128, 24, 64});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order_const);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose0, transpose1}, 2);
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
};

TEST_P(TransposeConcatTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(mlir_TransposeConcat, TransposeConcatTest, ::testing::Values(ov::element::f16), TransposeConcatTest::getTestCaseName);

}  // namespace
