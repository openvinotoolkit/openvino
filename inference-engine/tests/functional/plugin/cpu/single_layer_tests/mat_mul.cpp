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

    int calculateQuantizeInHigh(const int numMult, const int maxIn0 = 10, const int maxIn1 = 10) const {
        auto quantizeInHigh = maxIn0 * maxIn1;
        quantizeInHigh *= numMult;
        return quantizeInHigh;
    }

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
        if (nodeType == MatMulNodeType::FullyConnected || (nodeType == MatMulNodeType::MatMul && inShapeA.size() < 4 && inShapeB.size() < 4))
            std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        cpuNodeType = nodeType == MatMulNodeType::MatMul ? "MatMul" : "FullyConnected";

        if (nodeType == MatMulNodeType::MatMul) {
            if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
                inPrc = outPrc = netPrecision = Precision::BF16;
            else
                inPrc = outPrc = netPrecision;
        }

        auto transpose = [](SizeVector& shape) {
            IE_ASSERT(shape.size() > 1);
            std::swap(*(shape.end() - 1), *(shape.end() - 2));
        };

        if (transpA) transpose(inShapeA);
        if (transpB) transpose(inShapeB);

        auto elemTypeA = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto elemTypeB = (elemTypeA == element::u8) ? element::i8 : elemTypeA;

        auto params = builder::makeParams(elemTypeA, {inShapeA});
        auto matrixB = builder::makeInputLayer(elemTypeB, secondaryInputType, inShapeB);
        if (secondaryInputType == helpers::InputLayerType::PARAMETER) {
            params.push_back(std::dynamic_pointer_cast<opset1::Parameter>(matrixB));
        }
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));

        std::shared_ptr<ngraph::Node> matMul;
        if (nodeType == MatMulNodeType::MatMul) {
            matMul = builder::makeMatMul(paramOuts[0], matrixB, transpA, transpB);
        } else {
            matMul = builder::makeFullyConnectedRelaxed(paramOuts[0], matrixB, transpA, transpB);
        }

        if (outPrc == Precision::U8 || outPrc == Precision::I8) {
            threshold = 1.001f;
            abs_threshold = 1.001f;
            quantizeInHigh = calculateQuantizeInHigh(inShapeA[1]);
            outElemType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
        }

        if (inPrc == Precision::U8 || inPrc == Precision::I8) {
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::i8, element::f32>>());
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::u8, element::f32>>());
        }

        function = makeNgraphFunction(element::f32, params, matMul, cpuNodeType);
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
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };

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

    {{{7, 3}, false}, {{3, 5}, false}},
    {{{71, 3}, false}, {{3, 20}, false}},
};

std::vector<fusingSpecificParams> fusingParamsSet2D_FP32 {
        emptyFusingSpec,
        fusingBiasFC,
        fusingRelu,
        fusingMultiplyPerChannel,
        fusingScaleShift,
        fusingPReluPerChannel,
        fusingPReluPerTensor,
        fusingFakeQuantizePerChannelRelu,
        fusingFakeQuantizePerTensorRelu,
};

const auto fullyConnectedParams2D_FP32 = ::testing::Combine(::testing::ValuesIn(IS2D),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(Layout::ANY),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                       ::testing::Values(cpuEmptyPluginConfig));

const auto testParams2D_FP32 = ::testing::Combine(fullyConnectedParams2D_FP32,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_FP32));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_FP32, MatMulLayerCPUTest, testParams2D_FP32, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet2D_BF16 {
        emptyFusingSpec,
        fusingBiasFC,
        fusingRelu,
        fusingMultiplyPerChannel,
        fusingScaleShift,
};

const auto fullyConnectedParams2D_BF16 = ::testing::Combine(::testing::ValuesIn(IS2D),
                                                            ::testing::Values(Precision::BF16),
                                                            ::testing::Values(Precision::BF16),
                                                            ::testing::Values(Precision::BF16),
                                                            ::testing::Values(Layout::ANY),
                                                            ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                            ::testing::Values(cpuBF16PluginConfig));

const auto testParams2D_BF16 = ::testing::Combine(fullyConnectedParams2D_BF16,
                                                  ::testing::Values(MatMulNodeType::FullyConnected),
                                                  ::testing::ValuesIn(fusingParamsSet2D_BF16));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_BF16, MatMulLayerCPUTest, testParams2D_BF16, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet2D_I8 {
        emptyFusingSpec,
        fusingBiasFC,
        fusingRelu,
        fusingMultiplyPerChannel,
        fusingScaleShift,
        fusingPReluPerChannel,
        fusingPReluPerTensor,
};

const auto fullyConnectedParams2D_I8 = ::testing::Combine(::testing::ValuesIn(IS2D),
                                                          ::testing::Values(Precision::FP32),
                                                          ::testing::Values(Precision::U8, Precision::I8),
                                                          ::testing::Values(Precision::FP32, Precision::U8, Precision::I8),
                                                          ::testing::Values(Layout::ANY),
                                                          ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                          ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                          ::testing::Values(cpuEmptyPluginConfig));

const auto testParams2D_I8 = ::testing::Combine(fullyConnectedParams2D_I8,
                                                ::testing::Values(MatMulNodeType::FullyConnected),
                                                ::testing::ValuesIn(fusingParamsSet2D_I8));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_I8, MatMulLayerCPUTest, testParams2D_I8, MatMulLayerCPUTest::getTestCaseName);

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

const auto fullyConnectedParams3D_FP32 = ::testing::Combine(::testing::ValuesIn(IS3D),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(Precision::FP32),
                                                       ::testing::Values(Layout::ANY),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                       ::testing::Values(cpuEmptyPluginConfig));

const auto testParams3D_FP32 = ::testing::Combine(fullyConnectedParams3D_FP32,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_FP32, MatMulLayerCPUTest, testParams3D_FP32, MatMulLayerCPUTest::getTestCaseName);

const auto fullyConnectedParams3D_BF16 = ::testing::Combine(::testing::ValuesIn(IS3D),
                                                            ::testing::Values(Precision::BF16),
                                                            ::testing::Values(Precision::BF16),
                                                            ::testing::Values(Precision::BF16),
                                                            ::testing::Values(Layout::ANY),
                                                            ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                            ::testing::Values(cpuBF16PluginConfig));

const auto testParams3D_BF16 = ::testing::Combine(fullyConnectedParams3D_BF16,
                                                  ::testing::Values(MatMulNodeType::FullyConnected),
                                                  ::testing::ValuesIn(fusingParamsSet3D));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_BF16, MatMulLayerCPUTest, testParams3D_BF16, MatMulLayerCPUTest::getTestCaseName);

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
