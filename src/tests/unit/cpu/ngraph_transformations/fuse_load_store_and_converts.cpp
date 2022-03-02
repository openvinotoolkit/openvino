// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <subgraph_lowered.hpp>
#include <subgraph_customizable.hpp>
#include <snippets_helpers.hpp>
#include <ngraph_transformations/fuse_load_store_and_convert.hpp>
#include <common_test_utils/data_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

class FuseLoadAndStoreConvertTests : public TransformationTestsF {
public:
    void run() {
        ASSERT_TRUE(function);
        manager.register_pass<ov::intel_cpu::pass::FuseLoadConvert>();
        manager.register_pass<ov::intel_cpu::pass::FuseStoreConvert>();
    }
};
/*
TEST_F(FuseLoadAndStoreConvertTests, smoke_Snippets_SkipAfterInputsEltwise) {
        {
            auto data0 = std::make_shared<op::v0::Parameter>(ov::element::i8, ov::Shape{2, 3});
            const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(shape_size(ov::Shape{2, 3}), 0, 10.);
            auto const_data = std::make_shared<op::v0::Constant>(ov::element::i8, ov::Shape{2, 3}, const_values);
            auto add = std::make_shared<op::v1::Add>(data0, const_data);
            auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
            auto mul = std::make_shared<op::v1::Multiply>(add, sub);
            return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
        }
    const auto &f = EltwiseFunction({{2, 3}, {1, 3}});
    function = f.getOriginal();
    // None subgraphs are expected, since the whole graph is an eltwise chain after input
    function_ref = f.getOriginal();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipAfterInputsMatMulEltwise) {
    const auto &f = MatMulEltwiseBranchesFunction(std::vector<Shape> {{1, 3, 4, 4}, {1, 3, 4, 4}});
    function = f.getOriginal();
    // Fully tokenizable, since inputs are followed by MatMul
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvMulActivation) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Multiply>(),
                                                   std::make_shared<ov::op::v0::Tanh>(),
                                                   std::make_shared<ov::op::v0::Sqrt>()};
    std::vector<Shape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    function = f.getOriginal();
    // Fully tokenizable, since Mul with 2 inputs isn't fused into Convolution
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_SkipConvFused_ConvSumActivation) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Tanh>(),
                                                   std::make_shared<ov::op::v0::Sqrt>()};
    std::vector<Shape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    function = f.getOriginal();
    // Not tokenizable, since Add + Eltwises can be fused into Convolution
    function_ref = f.getOriginal();
    run();
}*/

}  // namespace snippets
}  // namespace test
}  // namespace ov