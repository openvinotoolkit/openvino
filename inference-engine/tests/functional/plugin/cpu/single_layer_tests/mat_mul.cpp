// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mat_mul.hpp"
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

using MatMulLayerCPUTestParamSet = std::tuple<MatMulLayerTestParamsSet,
                                              MatMulNodeType,
                                              fusingSpecificParams>;

class MatMulLayerCPUTest : public testing::WithParamInterface<MatMulLayerCPUTestParamSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerCPUTestParamSet>& obj) {
        MatMulLayerTestParamsSet basicParamsSet;
        MatMulNodeType nodeType;
        fusingSpecificParams fusingParams;

        std::tie(basicParamsSet, nodeType, fusingParams) = obj.param;

        std::ostringstream result;
        result << (nodeType == MatMulNodeType::MatMul ? "MatMul_" : "FullyConnected_");
        result << LayerTestsDefinitions::MatMulTest::getTestCaseName(
            testing::TestParamInfo<LayerTestsDefinitions::MatMulLayerTestParamsSet>(basicParamsSet, 0));
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
     std::string cpuNodeType;

    void SetUp() override {
        MatMulLayerTestParamsSet basicParamsSet;
        MatMulNodeType nodeType;
        fusingSpecificParams fusingParams;

        std::tie(basicParamsSet, nodeType, fusingParams) = this->GetParam();

        ShapeRelatedParams shapeRelatedParams;
        Precision netPrecision;
        helpers::InputLayerType secondaryInputType;
        std::map<std::string, std::string> additionalConfig;

        std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) = basicParamsSet;

        SizeVector inShapeA = shapeRelatedParams.input1.first;
        SizeVector inShapeB = shapeRelatedParams.input2.first;
        bool transpA = shapeRelatedParams.input1.second;
        bool transpB = shapeRelatedParams.input2.second;

        /* @todo
         * Currently nodes are not fused thought Reshape
         * Check can be deleted after this limitation is gone
         */
        if (nodeType == MatMulNodeType::MatMul && inShapeA.size() < 4 && inShapeB.size() < 4)
            std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            inPrc = outPrc = netPrecision = Precision::BF16;
        else
            inPrc = outPrc = netPrecision;

        cpuNodeType = nodeType == MatMulNodeType::MatMul ? "MatMul" : "FullyConnected";

        auto transpose = [](SizeVector& shape) {
            IE_ASSERT(shape.size() > 1);
            std::swap(*(shape.end() - 1), *(shape.end() - 2));
        };

        if (transpA) transpose(inShapeA);
        if (transpB) transpose(inShapeB);

        auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = builder::makeParams(ngPrec, {inShapeA});
        auto matrixB = builder::makeInputLayer(ngPrec, secondaryInputType, inShapeB);
        if (secondaryInputType == helpers::InputLayerType::PARAMETER) {
            params.push_back(std::dynamic_pointer_cast<opset1::Parameter>(matrixB));
        }
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));
        auto matMul = builder::makeMatMul(paramOuts[0], matrixB, transpA, transpB);
        function = makeNgraphFunction(ngPrec, params, matMul, cpuNodeType);
        functionRefs = ngraph::clone_function(*function);
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

std::vector<std::map<std::string, std::string>> additionalConfig {
    std::map<std::string, std::string>{/* empty config */},
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
};

const std::vector<Precision> netPRCs {
    Precision::FP32,
    Precision::BF16
};

/* ============= FullyConnected ============= */
namespace fullyConnected {

const auto fusingBiasFC = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<Node> inpNode, const element::Type& ngPrc, ParameterVector& params) {
                auto bias = builder::makeConstant(ngPrc, Shape({inpNode->get_output_shape(0).back()}), std::vector<float>{}, true);
                return std::make_shared<opset1::Add>(inpNode, bias);
            }, "fusingBiasFC"}}), {"Add"}};

const std::vector<ShapeRelatedParams> IS2D {
    {{{59, 1}, false}, {{1, 120}, false}},
    {{{59, 1}, true}, {{1, 120}, false}},
    {{{59, 1}, false}, {{1, 120}, true}},
    {{{59, 1}, true}, {{1, 120}, true}},

    {{{59, 120}, false}, {{120, 1}, false}},
    {{{59, 120}, true}, {{120, 1}, false}},
    {{{59, 120}, false}, {{120, 1}, true}},
    {{{59, 120}, true}, {{120, 1}, true}},

    {{{1, 120}, false}, {{120, 59}, false}},
    {{{1, 120}, true}, {{120, 59}, false}},
    {{{1, 120}, false}, {{120, 59}, true}},
    {{{1, 120}, true}, {{120, 59}, true}},

    {{{71, 128}, false}, {{128, 20}, false}},
    {{{71, 128}, true}, {{128, 20}, false}},
    {{{71, 128}, false}, {{128, 20}, true}},
    {{{71, 128}, true}, {{128, 20}, true}},
};

std::vector<fusingSpecificParams> fusingParamsSet2D {
        emptyFusingSpec,
        fusingBiasFC,
        fusingRelu,
        fusingMultiplyPerChannel,
        fusingPReluPerTensor
};

const auto fullyConnectedParams2D = ::testing::Combine(::testing::ValuesIn(IS2D),
                                                       ::testing::ValuesIn(netPRCs),
                                                       ::testing::Values(Precision::UNSPECIFIED),
                                                       ::testing::Values(Precision::UNSPECIFIED),
                                                       ::testing::Values(Layout::ANY),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                       ::testing::ValuesIn(additionalConfig));

const auto testParams2D = ::testing::Combine(fullyConnectedParams2D,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerCPUTest, testParams2D, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS3D = {
    {{{1, 32, 120}, false}, {{120, 5}, false}},
    {{{1, 32, 120}, true}, {{120, 5}, false}},
    {{{1, 32, 120}, false}, {{120, 5}, true}},
    {{{1, 32, 120}, true}, {{120, 5}, true}},

    {{{7, 32, 120}, false}, {{120, 50}, false}},
    {{{7, 32, 120}, true}, {{120, 50}, false}},
    {{{7, 32, 120}, false}, {{120, 50}, true}},
    {{{7, 32, 120}, true}, {{120, 50}, true}},
};

std::vector<fusingSpecificParams> fusingParamsSet3D {
        emptyFusingSpec,
        fusingBiasFC
};

const auto fullyConnectedParams3D = ::testing::Combine(::testing::ValuesIn(IS3D),
                                                       ::testing::ValuesIn(netPRCs),
                                                       ::testing::Values(Precision::UNSPECIFIED),
                                                       ::testing::Values(Precision::UNSPECIFIED),
                                                       ::testing::Values(Layout::ANY),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                       ::testing::ValuesIn(additionalConfig));

const auto testParams3D = ::testing::Combine(fullyConnectedParams3D,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerCPUTest, testParams3D, MatMulLayerCPUTest::getTestCaseName);

}; // namespace fullyConnected


/* ============= MatMul ============= */
namespace matmul {

const std::vector<ShapeRelatedParams> IS = {
    {{{1, 2, 32, 120}, false}, {{120, 5}, false}},
    {{{1, 2, 32, 120}, true}, {{120, 5}, false}},
    {{{1, 2, 32, 120}, false}, {{120, 5}, true}},
    {{{1, 2, 32, 120}, true}, {{120, 5}, true}},

    {{{7, 32, 120}, false}, {{3, 7, 120, 50}, false}},
    {{{7, 32, 120}, true}, {{3, 7, 120, 50}, false}},
    {{{7, 32, 120}, false}, {{3, 7, 120, 50}, true}},
    {{{7, 32, 120}, true}, {{3, 7, 120, 50}, true}},

    {{{10, 10, 10}, false}, {{10, 10, 10}, false}},
    {{{10, 10, 10}, true}, {{10, 10, 10}, false}},
    {{{10, 10, 10}, false}, {{10, 10, 10}, true}},
    {{{10, 10, 10}, true}, {{10, 10, 10}, true}},

    {{{55, 12}, false}, {{12, 55}, false}},
    {{{55, 12}, true}, {{12, 55}, false}},
    {{{55, 12}, false}, {{12, 55}, true}},
    {{{55, 12}, true}, {{12, 55}, true}},
};

std::vector<fusingSpecificParams> matmulFusingParams {
        emptyFusingSpec,
        fusingElu,
};

const auto matMulParams = ::testing::Combine(::testing::ValuesIn(IS),
                                             ::testing::ValuesIn(netPRCs),
                                             ::testing::Values(Precision::UNSPECIFIED),
                                             ::testing::Values(Precision::UNSPECIFIED),
                                             ::testing::Values(Layout::ANY),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParams = ::testing::Combine(matMulParams,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::ValuesIn(matmulFusingParams));

INSTANTIATE_TEST_SUITE_P(smoke_MM, MatMulLayerCPUTest, testParams, MatMulLayerCPUTest::getTestCaseName);

}; // namespace matmul

} // namespace

} // namespace CPULayerTestsDefinitions
