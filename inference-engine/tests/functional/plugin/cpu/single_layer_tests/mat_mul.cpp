// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

enum class MatMulNodeType {
    MatMul,
    FullyConnected
};

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ElementType,        // Network precision
        ElementType,        // Input precision
        ElementType,        // Output precision
        ngraph::helpers::InputLayerType,   // Secondary input type
        TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

using MatMulLayerCPUTestParamSet = std::tuple<MatMulLayerTestParamsSet,
                                              MatMulNodeType,
                                              fusingSpecificParams>;

class MatMulLayerCPUTest : public testing::WithParamInterface<MatMulLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerCPUTestParamSet>& obj) {
        MatMulLayerTestParamsSet basicParamsSet;
        MatMulNodeType nodeType;
        fusingSpecificParams fusingParams;

        std::tie(basicParamsSet, nodeType, fusingParams) = obj.param;

        ElementType netType;
        ElementType inType, outType;
        ShapeRelatedParams shapeRelatedParams;
        ngraph::helpers::InputLayerType secondaryInputType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapeRelatedParams, netType, inType, outType, secondaryInputType, targetDevice, additionalConfig) =
                basicParamsSet;

        std::ostringstream result;
        result << (nodeType == MatMulNodeType::MatMul ? "MatMul_" : "FullyConnected_");
        result << "IS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << CommonTestUtils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << shapeRelatedParams.transpose.first << "_";
        result << "transpose_b=" << shapeRelatedParams.transpose.second << "_";
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
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

    template<typename T>
    void transpose(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void SetUp() override {
        MatMulLayerTestParamsSet basicParamsSet;
        MatMulNodeType nodeType;
        fusingSpecificParams fusingParams;

        std::tie(basicParamsSet, nodeType, fusingParams) = this->GetParam();

        ShapeRelatedParams shapeRelatedParams;
        ElementType netType;
        helpers::InputLayerType secondaryInputType;
        std::map<std::string, std::string> additionalConfig;

        std::tie(shapeRelatedParams, netType, inType, outType, secondaryInputType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(shapeRelatedParams.inputShapes);

        bool transpA = shapeRelatedParams.transpose.first;
        bool transpB = shapeRelatedParams.transpose.second;

        if (transpA) {
            transpose(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[0]);
            }
        }
        if (transpB) {
            transpose(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        /* @todo
         * Currently nodes are not fused thought Reshape
         * Check can be deleted after this limitation is gone
         */
        if (nodeType == MatMulNodeType::MatMul && inShapeA.size() < 4 && inShapeB.size() < 4)
            std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            inType = outType = netType = ElementType::bf16;
        else
            inType = outType = netType;

        cpuNodeType = nodeType == MatMulNodeType::MatMul ? "MatMul" : "FullyConnected";
        selectedType = makeSelectedTypeStr("jit_gemm", outType);

        auto params = builder::makeDynamicParams(netType, {inShapeA});

        auto matrixB = builder::makeDynamicInputLayer(netType, secondaryInputType, inShapeB);
        if (secondaryInputType == helpers::InputLayerType::PARAMETER) {
            params.push_back(std::dynamic_pointer_cast<opset1::Parameter>(matrixB));
        }
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));
        auto matMul = builder::makeMatMul(paramOuts[0], matrixB, transpA, transpB);
        function = makeNgraphFunction(netType, params, matMul, cpuNodeType);
        checkFusingPosition = false;
    }
};

TEST_P(MatMulLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(executableNetwork, cpuNodeType);
}

namespace {

/* ============= Common params ============= */
std::vector<std::map<std::string, std::string>> additionalConfig {
    std::map<std::string, std::string>{/* empty config */},
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
};

const std::vector<ElementType> netPRCs {
    ElementType::f32,
    ElementType::bf16
};

/* ============= FullyConnected ============= */
namespace fullyConnected {

const auto fusingBiasFC = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<Node> inpNode, const element::Type& ngPrc, ParameterVector& params) {
                auto bias = builder::makeConstant(ngPrc, Shape({inpNode->get_output_shape(0).back()}), std::vector<float>{}, true);
                return std::make_shared<opset1::Add>(inpNode, bias);
            }, "fusingBiasFC"}}), {"Add"}};

const std::vector<ShapeRelatedParams> IS2D = {
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, false}},
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, false}},
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, true}},
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, true}},

    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, false}},
    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, false}},
    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, true}},
    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, true}},

    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, false}},
    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, false}},
    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, true}},
    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, true}},
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
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                       ::testing::ValuesIn(additionalConfig));

const auto testParams2D = ::testing::Combine(fullyConnectedParams2D,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerCPUTest, testParams2D, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS3D = {
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, true}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, true}},

    {static_shapes_to_test_representation({{1, 429}, {1, 429, 1}}), {true, true}}
};

std::vector<fusingSpecificParams> fusingParamsSet3D {
        emptyFusingSpec,
        fusingBiasFC
};

const auto fullyConnectedParams3D = ::testing::Combine(::testing::ValuesIn(IS3D),
                                                       ::testing::ValuesIn(netPRCs),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                       ::testing::ValuesIn(additionalConfig));

const auto testParams3D = ::testing::Combine(fullyConnectedParams3D,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerCPUTest, testParams3D, MatMulLayerCPUTest::getTestCaseName);

} // namespace fullyConnected


/* ============= MatMul ============= */
namespace matmul {

const std::vector<ShapeRelatedParams> IS = {
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, true}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, true}},

    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, true}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, true}},

    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}},

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },
};

std::vector<fusingSpecificParams> matmulFusingParams {
        emptyFusingSpec,
        fusingElu,
        fusingSqrt
};

const auto matMulParams = ::testing::Combine(::testing::ValuesIn(IS),
                                             ::testing::ValuesIn(netPRCs),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParams = ::testing::Combine(matMulParams,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::ValuesIn(matmulFusingParams));

INSTANTIATE_TEST_SUITE_P(smoke_MM, MatMulLayerCPUTest, testParams, MatMulLayerCPUTest::getTestCaseName);

} // namespace matmul

} // namespace

} // namespace CPULayerTestsDefinitions
