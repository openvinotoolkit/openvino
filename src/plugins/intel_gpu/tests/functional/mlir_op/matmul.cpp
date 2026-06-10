// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using BatchMatMulParams = std::tuple<ov::Shape,  // A shape
                                     ov::Shape,  // B shape
                                     bool,       // transpose A
                                     bool,       // transpose B
                                     ov::element::Type>;

class BatchMatMulTest : public testing::WithParamInterface<BatchMatMulParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchMatMulParams>& obj) {
        const auto& [a_shape, b_shape, tr_a, tr_b, prec] = obj.param;
        std::ostringstream result;
        result << "A=" << ov::test::utils::vec2str(a_shape) << "_";
        result << "B=" << ov::test::utils::vec2str(b_shape) << "_";
        result << "trA=" << tr_a << "_trB=" << tr_b << "_";
        result << "precision=" << prec;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [a_shape, b_shape, tr_a, tr_b, prec] = GetParam();
        auto param_a = std::make_shared<ov::op::v0::Parameter>(prec, a_shape);
        auto param_b = std::make_shared<ov::op::v0::Parameter>(prec, b_shape);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param_a, param_b, tr_a, tr_b);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b}, "BatchMatMul");
        abs_threshold = 100.f;
    }
};

TEST_P(BatchMatMulTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(mlir_BatchMatMul,
                         BatchMatMulTest,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}),
                                            ::testing::Values(ov::Shape{1536, 1536}),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::element::f16)),
                         BatchMatMulTest::getTestCaseName);

}  // namespace
