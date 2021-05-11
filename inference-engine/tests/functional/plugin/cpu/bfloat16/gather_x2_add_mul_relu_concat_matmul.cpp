// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class Gather_x2_add_mul_relu_concat_matmul : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//                       Add (FP32)
//                        |
//                     FullyConnected (BF16)
//                   /             |       \
// -------------------------------------------
//             Gather(FP32)  Gather(FP32)    Add (FP32)
//                 \           /              /
//                   Mul(FP32)     ReLU(FP32)
//                     \        /
//                       Concat(BF16)    Const
//                           \     /
//                           Matmul(BF16)

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        auto inputSize = inputShapes[1];

        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> addConst = nullptr;
        if (netPrecision == Precision::FP32) {
            addConst = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            addConst = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode0 = std::make_shared<opset1::Multiply>(input1, addConst);
        addNode0->set_friendly_name("Add_1");

        // matmul
        std::shared_ptr<ngraph::opset1::Constant> matmulConst0 = nullptr;
        if (netPrecision == Precision::FP32) {
            matmulConst0 = opset1::Constant::create(ntype, Shape{inputSize, inputSize}, { 2.0f });
        } else {
            matmulConst0 = opset1::Constant::create(ntype, Shape{inputSize, inputSize},
                    { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto matmulNode = std::make_shared<opset1::MatMul>(addNode0, matmulConst0);
        matmulNode->set_friendly_name("Matmul_0");

        // gather
        std::vector<size_t> gatherArray;
        for (size_t i = 0; i < inputSize; i++) {
            gatherArray.push_back(i);
        }
        auto axesConst = opset1::Constant::create(ngraph::element::i64, Shape{1}, { 1 });
        auto indexesConst = opset1::Constant::create(ngraph::element::i64, Shape{inputSize}, gatherArray);
        auto gatherNode1 = std::make_shared<opset1::Gather>(matmulNode, indexesConst, axesConst);
        gatherNode1->set_friendly_name("Gather_1");

        auto gatherNode2 = std::make_shared<opset1::Gather>(matmulNode, indexesConst, axesConst);
        gatherNode2->set_friendly_name("Gather_2");

        // multiply
        auto mulNode = std::make_shared<opset1::Multiply>(gatherNode1, gatherNode2);
        mulNode->set_friendly_name("Mul_1");

        // add
        auto addNode1 = std::make_shared<opset1::Multiply>(matmulNode, addConst);
        addNode0->set_friendly_name("Add_1");

        // ReLU
        auto reluNode =  std::make_shared<opset1::Relu>(addNode1);
        reluNode->set_friendly_name("Relu_1");

        // Concat
        ngraph::NodeVector concInputNodes = {mulNode, reluNode};
        auto concNode = std::make_shared<opset1::Concat>(concInputNodes, 1);
        concNode->set_friendly_name("Conc_1");

        // matmul
        std::shared_ptr<ngraph::opset1::Constant> matmulConst1 = nullptr;
        if (netPrecision == Precision::FP32) {
            matmulConst1 = opset1::Constant::create(ntype, Shape{inputSize * 2, inputSize * 2}, { 2.0f });
        } else {
            matmulConst1 = opset1::Constant::create(ntype, Shape{inputSize * 2, inputSize * 2},
                    { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto matmulNode1 = std::make_shared<opset1::MatMul>(concNode, matmulConst1);
        matmulNode1->set_friendly_name("Matmul_1");

        return std::make_shared<ngraph::Function>(matmulNode1, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2: set up safe threshold <= 5% from maximum value of output tensor
        threshold = 177.f;  // Max in fp32 network by output:  3887.11

        // STAGE3:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Matmul_0"] = "BF16";
        expectedPrecisions["Mul_1"] = "BF16";
        expectedPrecisions["Add_1"] = netPrecision.name(); // FP32->BF16 in case of FP32 net, BF16->BF16 in case of BF16 net
        expectedPrecisions["Relu_1"] = "ndef";
        expectedPrecisions["Conc_1"] = "BF16";
        expectedPrecisions["Matmul_1"] = "BF16";
    }
};

TEST_P(Gather_x2_add_mul_relu_concat_matmul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};


INSTANTIATE_TEST_CASE_P(smoke_FP32_bfloat16_NoReshape, Gather_x2_add_mul_relu_concat_matmul,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({ 2048, 64 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Gather_x2_add_mul_relu_concat_matmul::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, Gather_x2_add_mul_relu_concat_matmul,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({ 2048, 64 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Gather_x2_add_mul_relu_concat_matmul::getTestCaseName);

}  // namespace LayerTestsDefinitions
