// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace {
// Input and Weights shapes
typedef std::pair<ov::Shape, ov::Shape> MatMulParams;

class MatMulSumFusion : public testing::WithParamInterface<MatMulParams>,
                        virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulParams>& params) {
        std::pair<ov::Shape, ov::Shape> shapes;
        shapes = params.param;

        std::ostringstream result;
        const char separator = '_';
        result << "IS=(";
        result << ov::test::utils::partialShape2str({shapes.first}) << separator;
        result << ov::test::utils::partialShape2str({shapes.second}) << separator;
        result << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = ov::element::f16;
        ov::Shape matmul_input_shape;
        ov::Shape weights_shape;
        std::pair<ov::Shape, ov::Shape> shapes;
        shapes = GetParam();
        matmul_input_shape = shapes.first;
        weights_shape = shapes.second;

        auto matmul_input = std::make_shared<ov::op::v0::Parameter>(type, matmul_input_shape);
        auto weights = ov::op::v0::Constant::create(type, weights_shape, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_input, weights, false, true);
        auto sum_input = std::make_shared<ov::op::v0::Parameter>(type, matmul->get_output_shape(0));
        auto activation = std::make_shared<ov::op::v0::Relu>(sum_input);
        auto sum = std::make_shared<ov::op::v1::Add>(matmul, activation);

        function = std::make_shared<ov::Model>(ov::OutputVector{sum}, ov::ParameterVector{matmul_input, sum_input});
    }
};

TEST_P(MatMulSumFusion, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU_MatMul, MatMulSumFusion,
                            ::testing::Values(std::make_pair(ov::Shape{2, 32}, ov::Shape{8, 32}),
                                              std::make_pair(ov::Shape{10, 17}, ov::Shape{5, 17})),
                         MatMulSumFusion::getTestCaseName);

}  // namespace
