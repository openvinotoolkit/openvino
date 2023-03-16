// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/common_utils.hpp"

#include <algorithm>
#include <cassert>

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using ElementType = ov::element::Type_t;
using MatmulBrgemmInt8TestParams = std::tuple<SizeVector,       // input shape
                                              bool,             // true: FullyConnected false: Matmul
                                              ElementType,      // quant to u8/s8
                                              fusingSpecificParams>;

// subgraph:
//   fq->MatMul/FullyConnected->[fq]
// can cover brgemm avx2:
//   (u8/s8 + s8)->f32
//   (u8/s8 + s8)->u8
class MatmulBrgemmInt8Test : public testing::WithParamInterface<MatmulBrgemmInt8TestParams>, public CpuTestWithFusing,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulBrgemmInt8TestParams> obj) {
        SizeVector supportedInputShapes;
        bool isFC;
        ElementType inType;
        fusingSpecificParams fusingParams;
        std::tie(supportedInputShapes, isFC, inType, fusingParams) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(supportedInputShapes) << "_";
        result << (isFC ? "FullyConnected" : "MatMul") << "_";
        result << "InputType=" << inType;
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    bool isFC;
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        SizeVector inShapes;
        ElementType inType;
        fusingSpecificParams fusingParams;
        std::tie(inShapes, isFC, inType, fusingParams) = this->GetParam();

        std::tie(postOpMgrPtr, fusedOps) = fusingParams;
        const auto ngPrec = element::f32;
        auto inputParams = builder::makeParams(ngPrec, {inShapes});

        std::shared_ptr<Node> fq;
        std::shared_ptr<Node> matMul;
        std::shared_ptr<Node> lastNode;
        if (inType == ElementType::u8)
            fq = ngraph::builder::makeFakeQuantize(inputParams[0], ngPrec, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        else
            fq = ngraph::builder::makeFakeQuantize(inputParams[0], ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

        if (isFC) {
            ngraph::Shape weightShape = inShapes;
            std::swap(weightShape[0], weightShape[1]);
            auto weightsNode = ngraph::builder::makeConstant(ngPrec, weightShape, std::vector<float>{0.0f}, true);
            auto fq2 = ngraph::builder::makeFakeQuantize(weightsNode, ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
            matMul = std::make_shared<ngraph::opset1::MatMul>(fq, fq2, false, false);

            auto biasWeightsNode = ngraph::builder::makeConstant(ngPrec, {}, std::vector<float>{0.0f}, true);
            lastNode = std::make_shared<ngraph::opset1::Add>(matMul, biasWeightsNode);
        } else {
            auto fq2 = ngraph::builder::makeFakeQuantize(inputParams[0], ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
            matMul = builder::makeMatMul(fq, fq2, false, true);
            lastNode = matMul;
        }
        matMul->get_rt_info() = CPUTestsBase::makeCPUInfo({}, {}, {"brgemm_avx2"});

        selectedType = makeSelectedTypeStr("brgemm_avx2", ElementType::i8);

        function = makeNgraphFunction(ngPrec, inputParams, lastNode, "MatmulBrgemmInt8");
    }
};

TEST_P(MatmulBrgemmInt8Test, CompareWithRefs) {
    // only cover avx2
    if (InferenceEngine::with_cpu_x86_avx512_core() || !InferenceEngine::with_cpu_x86_avx2_vnni())
        GTEST_SKIP();

    Run();
    CheckPluginRelatedResults(executableNetwork, isFC ? "FullyConnected" : "MatMul");
}

namespace {

const std::vector<SizeVector> supportedInputShapes = {
    {16, 32},
    {17, 15},
};

// verify fusing just in case
std::vector<fusingSpecificParams> fusingParamsSet {
    emptyFusingSpec,
    fusingFakeQuantizePerChannel,
};

INSTANTIATE_TEST_SUITE_P(smoke_matmulBrgemmInt8, MatmulBrgemmInt8Test,
                         ::testing::Combine(::testing::ValuesIn(supportedInputShapes),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({ElementType::u8, ElementType::i8}),
                                            ::testing::ValuesIn(fusingParamsSet)),
                         MatmulBrgemmInt8Test::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
