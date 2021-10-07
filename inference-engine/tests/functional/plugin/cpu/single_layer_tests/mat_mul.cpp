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

using ShapesDefenition = std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>;

struct ShapeRelatedParams {
    ShapesDefenition inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        ngraph::helpers::InputLayerType,   // Secondary input type
        LayerTestsUtils::TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

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

        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout;
        ShapeRelatedParams shapeRelatedParams;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::string targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
                basicParamsSet;

        std::ostringstream result;
        result << (nodeType == MatMulNodeType::MatMul ? "MatMul_" : "FullyConnected_");
        if (!shapeRelatedParams.inputShapes.first.empty()) {
            result << "IS=" << CommonTestUtils::partialShape2str(shapeRelatedParams.inputShapes.first) << "_";
        }
        result << "TS=";
        for (const auto& shape : shapeRelatedParams.inputShapes.second) {
            result << "(";
            if (!shape.empty()) {
                auto itr = shape.begin();
                do {
                    result << CommonTestUtils::vec2str(*itr);
                } while (++itr != shape.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << shapeRelatedParams.transpose.first << "_";
        result << "transpose_b=" << shapeRelatedParams.transpose.second << "_";
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "trgDev=" << targetDevice;
        result << "config=(";
        for (const auto configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";
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

        bool transpA = shapeRelatedParams.transpose.first;
        bool transpB = shapeRelatedParams.transpose.second;

        auto transpose = [](ngraph::Shape& shape) {
            IE_ASSERT(shape.size() > 1);
            std::swap(*(shape.end() - 1), *(shape.end() - 2));
        };

        if (transpA) {
            for (auto& item : shapeRelatedParams.inputShapes.second) {
                transpose(item[0]);
            }
        }
        if (transpB) {
            for (auto& item : shapeRelatedParams.inputShapes.second) {
                transpose(item[1]);
            }
        }

        SizeVector inShapeA = shapeRelatedParams.inputShapes.second[0][0];
        SizeVector inShapeB = shapeRelatedParams.inputShapes.second[0][1];

        inputDynamicShapes = shapeRelatedParams.inputShapes.first;
        targetStaticShapes = shapeRelatedParams.inputShapes.second;

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

        auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = builder::makeParams(ngPrec, {inShapeA});
        auto matrixB = builder::makeInputLayer(ngPrec, secondaryInputType, inShapeB);
        if (secondaryInputType == helpers::InputLayerType::PARAMETER) {
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

const std::vector<ShapeRelatedParams> IS2D = {
    {{{}, {{{59, 1}, {1, 120}}}}, {false, false}},
    {{{}, {{{59, 1}, {1, 120}}}}, {true, false}},
    {{{}, {{{59, 1}, {1, 120}}}}, {false, true}},
    {{{}, {{{59, 1}, {1, 120}}}}, {true, true}},

    {{{}, {{{59, 120}, {120, 1}}}}, {false, false}},
    {{{}, {{{59, 120}, {120, 1}}}}, {true, false}},
    {{{}, {{{59, 120}, {120, 1}}}}, {false, true}},
    {{{}, {{{59, 120}, {120, 1}}}}, {true, true}},

    {{{}, {{{1, 120}, {120, 59}}}}, {false, false}},
    {{{}, {{{1, 120}, {120, 59}}}}, {true, false}},
    {{{}, {{{1, 120}, {120, 59}}}}, {false, true}},
    {{{}, {{{1, 120}, {120, 59}}}}, {true, true}},

    {{{}, {{{71, 128}, {128, 20}}}}, {false, false}},
    {{{}, {{{71, 128}, {128, 20}}}}, {true, false}},
    {{{}, {{{71, 128}, {128, 20}}}}, {false, true}},
    {{{}, {{{71, 128}, {128, 20}}}}, {true, true}},
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
    {{{}, {{{1, 32, 120}, {120, 5}}}}, {false, false}},
    {{{}, {{{1, 32, 120}, {120, 5}}}}, {true, false}},
    {{{}, {{{1, 32, 120}, {120, 5}}}}, {false, true}},
    {{{}, {{{1, 32, 120}, {120, 5}}}}, {true, true}},

    {{{}, {{{1, 32, 120}, {120, 50}}}}, {false, false}},
    {{{}, {{{1, 32, 120}, {120, 50}}}}, {true, false}},
    {{{}, {{{1, 32, 120}, {120, 50}}}}, {false, true}},
    {{{}, {{{1, 32, 120}, {120, 50}}}}, {true, true}},
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
    {{{}, {{{1, 2, 32, 120}, {120, 5}}}}, {false, false}},
    {{{}, {{{1, 2, 32, 120}, {120, 5}}}}, {true, false}},
    {{{}, {{{1, 2, 32, 120}, {120, 5}}}}, {false, true}},
    {{{}, {{{1, 2, 32, 120}, {120, 5}}}}, {true, true}},

    {{{}, {{{7, 32, 120}, {3, 7, 120, 50}}}}, {false, false}},
    {{{}, {{{7, 32, 120}, {3, 7, 120, 50}}}}, {true, false}},
    {{{}, {{{7, 32, 120}, {3, 7, 120, 50}}}}, {false, true}},
    {{{}, {{{7, 32, 120}, {3, 7, 120, 50}}}}, {true, true}},

    {{{}, {{{10, 10, 10}, {10, 10, 10}}}}, {false, false}},
    {{{}, {{{10, 10, 10}, {10, 10, 10}}}}, {true, false}},
    {{{}, {{{10, 10, 10}, {10, 10, 10}}}}, {false, true}},
    {{{}, {{{10, 10, 10}, {10, 10, 10}}}}, {true, true}},

    {{{}, {{{55, 12}, {12, 55}}}}, {false, false}},
    {{{}, {{{55, 12}, {12, 55}}}}, {true, false}},
    {{{}, {{{55, 12}, {12, 55}}}}, {false, true}},
    {{{}, {{{55, 12}, {12, 55}}}}, {true, true}},

    {{{{-1, -1}, {-1, -1}}, {{{55, 12}, {12, 55}}, {{33, 7}, {7, 33}}}}, {false, false}},
    {{{{-1, -1}, {-1, -1}}, {{{55, 12}, {12, 55}}, {{33, 7}, {7, 33}}}}, {true, false}},
    {{{{-1, -1}, {-1, -1}}, {{{55, 12}, {12, 55}}, {{33, 7}, {7, 33}}}}, {false, true}},
    {{{{-1, -1}, {-1, -1}}, {{{55, 12}, {12, 55}}, {{33, 7}, {7, 33}}}}, {true, true}},

    {{{{-1, -1, -1, -1}, {-1, -1}}, {{{1, 2, 32, 60}, {60, 5}}, {{1, 2, 32, 30}, {30, 5}}}}, {false, false}},
    {{{{-1, -1, -1, -1}, {-1, -1}}, {{{1, 2, 32, 60}, {60, 5}}, {{1, 2, 32, 30}, {30, 5}}}}, {true, false}},
    {{{{-1, -1, -1, -1}, {-1, -1}}, {{{1, 2, 32, 60}, {60, 5}}, {{1, 2, 32, 30}, {30, 5}}}}, {false, true}},
    {{{{-1, -1, -1, -1}, {-1, -1}}, {{{1, 2, 32, 60}, {60, 5}}, {{1, 2, 32, 30}, {30, 5}}}}, {true, true}},

    {{{{-1, -1, -1}, {-1, -1, -1, -1}}, {{{7, 32, 60}, {3, 7, 60, 25}}, {{7, 32, 30}, {3, 7, 30, 25}}}}, {false, false}},
    {{{{-1, -1, -1}, {-1, -1, -1, -1}}, {{{7, 32, 60}, {3, 7, 60, 25}}, {{7, 32, 30}, {3, 7, 30, 25}}}}, {true, false}},
    {{{{-1, -1, -1}, {-1, -1, -1, -1}}, {{{7, 32, 60}, {3, 7, 60, 25}}, {{7, 32, 30}, {3, 7, 30, 25}}}}, {false, true}},
    {{{{-1, -1, -1}, {-1, -1, -1, -1}}, {{{7, 32, 60}, {3, 7, 60, 25}}, {{7, 32, 30}, {3, 7, 30, 25}}}}, {true, true}},

    {{{{-1, -1, -1}, {-1, -1, -1}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {false, false}},
    {{{{-1, -1, -1}, {-1, -1, -1}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {true, false}},
    {{{{-1, -1, -1}, {-1, -1, -1}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {false, true}},
    {{{{-1, -1, -1}, {-1, -1, -1}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {true, true}},

    {{{{{1, 15}, {1, 15}, {1, 15}}, {{1, 15}, {1, 15}, {1, 15}}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {false, false}},
    {{{{{1, 15}, {1, 15}, {1, 15}}, {{1, 15}, {1, 15}, {1, 15}}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {true, false}},
    {{{{{1, 15}, {1, 15}, {1, 15}}, {{1, 15}, {1, 15}, {1, 15}}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {false, true}},
    {{{{{1, 15}, {1, 15}, {1, 15}}, {{1, 15}, {1, 15}, {1, 15}}}, {{{10, 10, 10}, {10, 10, 10}}, {{5, 5, 5}, {5, 5, 5}}}}, {true, true}},
};

std::vector<fusingSpecificParams> matmulFusingParams {
        emptyFusingSpec,
        fusingElu,
        fusingSqrt
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
