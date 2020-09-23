// Copyright (C) 2020 Intel Corporation
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

class Fc_gather : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//                   Add (FP32)
//                    |
//                  FC (BF16)
//                    |
// -------------------------------------------
//                Gather(BF16)  Const
//                 \           /
//                   Add(FP32)

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto inputSize = inputShapes[1];

        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});

        input1->set_friendly_name("Input_1");
        size_t gatherDecrementSize = 1;
        std::shared_ptr<ngraph::opset1::Constant> addConst = nullptr;
        if (netPrecision == Precision::FP32) {
            addConst = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            addConst = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode0 = std::make_shared<opset1::Multiply>(input1, addConst);
        addNode0->set_friendly_name("Add_0");

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
        auto axesConst = opset1::Constant::create(ngraph::element::i64, Shape{1}, { 1 });
        std::vector<size_t> gatherArray;
        for (size_t i = 0; i < inputSize - gatherDecrementSize; i++) {
            gatherArray.push_back(i);
        }
        auto indexesConst = opset1::Constant::create(ngraph::element::i64, Shape{inputSize - gatherDecrementSize}, gatherArray);
        auto gatherNode = std::make_shared<opset1::Gather>(matmulNode, indexesConst, axesConst);
        gatherNode->set_friendly_name("Gather_1");

        auto addNode1 = std::make_shared<opset1::Multiply>(gatherNode, addConst);
        addNode1->set_friendly_name("Add_1");

        return std::make_shared<ngraph::Function>(addNode1, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2: set up safe threshold <= 5% from maximum value of output tensor
        threshold = 0.1f;  // Max in fp32 network by output:  21.7285

        // STAGE3:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters

        expectedPrecisions["Gather_1"] = "BF16";
    }
};

TEST_P(Fc_gather, CompareWithRefImpl) {
    test();
};


INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, Fc_gather,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({1, 4})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Fc_gather::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BF16_bfloat16_NoReshape, Fc_gather,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({1, 4})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Fc_gather::getTestCaseName);

}  // namespace LayerTestsDefinitions
