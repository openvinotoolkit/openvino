// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/normalize_l2.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

enum class MatMulNodeType {
    MatMul,
    FullyConnected
};

using MatMulLayerTestParams = std::tuple<std::pair<SizeVector, SizeVector>,
                                         Precision,
                                         helpers::InputLayerType,
                                         bool,
                                         bool>;

using MatMulLayerCPUTestParamSet = std::tuple<MatMulLayerTestParams,
                                              MatMulNodeType,
                                              fusingSpecificParams>;

class MatMulLayerCPUTest : public testing::WithParamInterface<MatMulLayerCPUTestParamSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulLayerCPUTestParamSet> obj) {
        MatMulLayerTestParams basicParamsSet;
        fusingSpecificParams fusingParams;
        MatMulNodeType nodeType;
        std::tie(basicParamsSet, nodeType, fusingParams) = obj.param;

        std::pair<SizeVector, SizeVector> IS;
        SizeVector isA, isB;
        bool transpA, transpB;
        Precision prec;
        helpers::InputLayerType typeB;
        std::tie(IS, prec, typeB, transpA, transpB) = basicParamsSet;
        isA = IS.first; isB = IS.second;

        std::ostringstream result;
        result << (nodeType == MatMulNodeType::MatMul ? "MatMul_" : "FullyConnected_");
        result << "IS_A=" << CommonTestUtils::vec2str(isA) << "_";
        result << "IS_B=" << CommonTestUtils::vec2str(isB) << "_";
        result << "Transp_A=" << transpA << "_";
        result << "Transp_B=" << transpB << "_";
        result << "Prec=" << prec << "_";
        result << "typeB=" << typeB;

        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
     std::string cpuNodeType;

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        MatMulLayerTestParams basicParamsSet;
        MatMulNodeType nodeType;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, nodeType, fusingParams) = this->GetParam();

        cpuNodeType = nodeType == MatMulNodeType::MatMul ? "MatMul" : "FullyConnected";

        std::pair<SizeVector, SizeVector> IS;
        SizeVector isA, isB;
        bool transpA, transpB;
        Precision prec;
        helpers::InputLayerType typeB;
        std::tie(IS, prec, typeB, transpA, transpB) = basicParamsSet;

        isA = IS.first; isB = IS.second;
        if (transpA) {
            IE_ASSERT(isA.size() > 1);
            std::swap(*(isA.end() - 1), *(isA.end() - 2));
        }
        if (transpB) {
            IE_ASSERT(isB.size() > 1);
            std::swap(*(isB.end() - 1), *(isB.end() - 2));
        }

        auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prec);
        auto params = builder::makeParams(ngPrec, {isA});
        auto matrixB = builder::makeInputLayer(ngPrec, typeB, isB);
        if (typeB == helpers::InputLayerType::PARAMETER) {
            params.push_back(std::dynamic_pointer_cast<opset1::Parameter>(matrixB));
        }
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));
        auto matMul = builder::makeMatMul(paramOuts[0], matrixB, transpA, transpB);
        function = makeNgraphFunction(ngPrec, params, matMul, cpuNodeType);
        checkFusingPosition = false;
    }
};

TEST_P(MatMulLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckFusingResults(executableNetwork, cpuNodeType);
}

namespace {

/* ============= Common params ============= */
const std::vector<bool> transpose = {
    true, false
};

/* ============= FullyConnected ============= */
namespace fullyConnected {

const auto fusingBiasFC = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<Node> inpNode, const element::Type& ngPrc, ParameterVector& params) {
                auto bias = builder::makeConstant(ngPrc, Shape({inpNode->get_input_shape(1).back()}), std::vector<float>{}, true);
                return std::make_shared<opset1::Add>(inpNode, bias);
            }, "fusingBiasFC"}}), {"Add"}};

const std::vector<std::pair<SizeVector, SizeVector>> IS2D = {
    {{59, 1}, {1, 120}},
    {{59, 120}, {120, 1}},
    {{1, 120}, {120, 59}},
    {{71, 128}, {128, 20}}
};

std::vector<fusingSpecificParams> fusingParamsSet2D {
        emptyFusingSpec,
        fusingBiasFC,
        fusingRelu,
        fusingMultiplyPerChannel,
        fusingPReluPerTensor
};

const auto fullyConnectedParams2D = ::testing::Combine(::testing::ValuesIn(IS2D),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::ValuesIn(transpose),
                                                       ::testing::ValuesIn(transpose));

const auto testParams2D = ::testing::Combine(fullyConnectedParams2D,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D));

INSTANTIATE_TEST_SUITE_P(smoke_Check_2D, MatMulLayerCPUTest, testParams2D, MatMulLayerCPUTest::getTestCaseName);

const std::vector<std::pair<SizeVector, SizeVector>> IS3D = {
    {{1, 32, 120}, {120, 5}},
    {{7, 32, 120}, {120, 50}}
};

std::vector<fusingSpecificParams> fusingParamsSet3D {
        emptyFusingSpec,
        fusingBiasFC
};

const auto fullyConnectedParams3D = ::testing::Combine(::testing::ValuesIn(IS3D),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::ValuesIn(transpose),
                                                       ::testing::ValuesIn(transpose));

const auto testParams3D = ::testing::Combine(fullyConnectedParams3D,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D));

INSTANTIATE_TEST_SUITE_P(smoke_Check_3D, MatMulLayerCPUTest, testParams3D, MatMulLayerCPUTest::getTestCaseName);

}; // namespace fullyConnected


/* ============= Gemm ============= */
namespace gemm {

const std::vector<std::pair<SizeVector, SizeVector>> IS = {
    {{1, 2, 32, 120}, {120, 5}},
    {{7, 32, 120}, {3, 7, 120, 50}},
    {{10, 10, 10}, {10, 10, 10}},
    {{55, 12}, {12, 55}}
};

const auto gemmParams = ::testing::Combine(::testing::ValuesIn(IS),
                                           ::testing::Values(Precision::FP32),
                                           ::testing::Values(helpers::InputLayerType::PARAMETER),
                                           ::testing::ValuesIn(transpose),
                                           ::testing::ValuesIn(transpose));

const auto testParams = ::testing::Combine(gemmParams,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_Check, MatMulLayerCPUTest, testParams, MatMulLayerCPUTest::getTestCaseName);

}; // namespace gemm

} // namespace

} // namespace CPULayerTestsDefinitions
