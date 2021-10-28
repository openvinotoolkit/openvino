// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using BroadcastLayerTestParamsSet = typename std::tuple<
        inputShapesPair,                       // Shapes
        std::vector<int64_t>,                  // Target shapes
        std::vector<int64_t>,                  // Axes mapping
        ov::op::BroadcastType,                 // Broadcast mode
        InferenceEngine::Precision,            // Network precision
        std::vector<bool>,                     // Const inputs
        std::string>;                          // Device name

using BroadcastLayerCPUTestParamsSet = typename std::tuple<
        BroadcastLayerTestParamsSet,
        CPUSpecificParams>;

class BroadcastLayerCPUTest : public testing::WithParamInterface<BroadcastLayerCPUTestParamsSet>,
                              virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastLayerCPUTestParamsSet> obj) {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        inputShapesPair inputShapes;
        std::vector<int64_t> targetShapes, axesMapping;
        ov::op::BroadcastType mode;
        InferenceEngine::Precision netPrecision;
        std::vector<bool> isConstInputs;
        std::string deviceName;
        std::tie(inputShapes, targetShapes, axesMapping, mode, netPrecision, isConstInputs, deviceName) = basicParamsSet;

        std::ostringstream result;
        result << "DynShapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        result << "StatShapes=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
        result << "targetShape=" << CommonTestUtils::vec2str(targetShapes)  << "_";
        result << "axesMapping=" << CommonTestUtils::vec2str(axesMapping)  << "_";
        result << "mode=" << mode << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "constIn=(" << (isConstInputs[0] ? "True" : "False") << "." << (isConstInputs[1] ? "True" : "False") << ")_";
        result << "trgDev=" << deviceName;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        inputShapesPair inputShapes;
        ov::op::BroadcastType mode;
        InferenceEngine::Precision netPrecision;
        std::vector<bool> isConstInput;
        std::tie(inputShapes, targetShape, axesMapping, mode, netPrecision, isConstInput, targetDevice) = basicParamsSet;
        bool isTargetShapeConst = isConstInput[0], isAxesMapConst = isConstInput[1];
        const auto targetShapeRank = targetShape.size();
        const auto axesMappingRank = axesMapping.size();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType += std::string("_") + netPrecision.name();

        const size_t dynShapesNum = std::min(inputShapes.first.size(), 3lu - isTargetShapeConst - isAxesMapConst);
        for (size_t i = 0lu; i < dynShapesNum; i++) {
            inputDynamicShapes.push_back(inputShapes.first[i]);
            if (!isTargetShapeConst) {
                inputDynamicShapes.push_back({ static_cast<int64_t>(targetShape.size()) });
            }
            if (!isAxesMapConst) {
                inputDynamicShapes.push_back({ static_cast<int64_t>(axesMapping.size()) });
            }
        }
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            targetStaticShapes.push_back({ inputShapes.second[i] });
            if (!isTargetShapeConst)
                targetStaticShapes[i].push_back({ targetShape.size() });
            if (!isAxesMapConst)
                targetStaticShapes[i].push_back({ axesMapping.size() });
        }

        ov::Shape inputDataShape = targetStaticShapes.front().front();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector functionParams = ngraph::builder::makeParams(ngPrc, { {"data", inputDataShape} });
        if (!isTargetShapeConst) {
            functionParams.push_back(ngraph::builder::makeParams(ov::element::i32, { {"targetShape", {targetShapeRank}} })[0]);
        }
        if (!isAxesMapConst) {
            functionParams.push_back(ngraph::builder::makeParams(ov::element::i32, { {"axesMapping", {axesMappingRank}} })[0]);
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));

        std::shared_ptr<ov::op::v3::Broadcast> broadcastOp;
        if (mode == ov::op::BroadcastType::EXPLICIT) {
            std::shared_ptr<ov::Node> targetShapeOp;
            std::shared_ptr<ov::Node> axesMappingOp;
            if (isTargetShapeConst) {
                targetShapeOp = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
            } else {
                targetShapeOp = functionParams[0];
            }
            if (isAxesMapConst) {
                axesMappingOp = ov::op::v0::Constant::create(ov::element::i64, {axesMappingRank}, axesMapping);
            } else {
                axesMappingOp = functionParams.size() > 2 ? functionParams[2] : functionParams[1];
            }
            broadcastOp = std::make_shared<ov::op::v3::Broadcast>(paramOuts[0],
                                                               targetShapeOp,
                                                               axesMappingOp,
                                                               mode);
        } else if (mode == ov::op::BroadcastType::NUMPY) {
            if (isTargetShapeConst) {
                auto targetShapeConst = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(paramOuts[0],
                                                                      targetShapeConst,
                                                                      mode);
            } else {
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(paramOuts[0],
                                                                      paramOuts[1],
                                                                      mode);
            }
        }

        broadcastOp->get_rt_info() = getCPUInfo();
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(broadcastOp)};
        function = std::make_shared<ov::Function>(results, functionParams, "Broadcast");
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &inputInfo) const override {
        if (inputInfo.name() == "targetShape") {
            const auto& td = inputInfo.getTensorDesc();
            return FuncTestUtils::createAndFillBlobWithFloatArray<int64_t>(td, targetShape.data(), targetShape.size());
        } else if (inputInfo.name() == "axesMapping") {
            const auto& td = inputInfo.getTensorDesc();
            return FuncTestUtils::createAndFillBlobWithFloatArray<int64_t>(td, axesMapping.data(), axesMapping.size());
        } else {
            return LayerTestsCommon::GenerateInput(inputInfo);
        }
    }

    std::vector<int64_t> targetShape, axesMapping;
};

TEST_P(BroadcastLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Broadcast");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* COMMON PARAMS */
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I8
};
/* ============= */

/* INSTANCES */
// 4D
const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc
};

const std::vector<inputShapesPair> staticInputShapes4D = {
    {
        {},
        { // Static shapes
            {{1, 16, 1, 1}}
        }
    },
    {
        {},
        { // Static shapes
            {{50, 50}}
        }
    }
};

INSTANTIATE_TEST_CASE_P(smoke_StaticShape4D, BroadcastLayerCPUTest,
                    ::testing::Combine(
                            ::testing::Combine(
                            ::testing::Values(staticInputShapes4D[0]),
                            ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 16, 3, 3}, {1, 16, 1, 3}}),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::Values(ov::op::BroadcastType::NUMPY),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<bool>{true, true}),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::ValuesIn(CPUParams4D)),
                    BroadcastLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_StaticShape4DE, BroadcastLayerCPUTest,
                    ::testing::Combine(
                        ::testing::Combine(
                            ::testing::Values(staticInputShapes4D[1]),
                            ::testing::Values(std::vector<int64_t>{1, 50, 50, 16}),
                            ::testing::Values(std::vector<int64_t>{1, 2}),
                            ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<bool>{true, true}),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                    BroadcastLayerCPUTest::getTestCaseName);
const std::vector<inputShapesPair> dynamicInputShapes4D = {
    {
        { // Origin dynamic shapes
            {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)}
        },
        { // Dynamic shapes instances
            {{1, 16, 1, 1}},
            {{8, 1, 1, 7}},
            {{1, 1, 1, 7}}
        }
    },
    {
        { // Origin dynamic shapes
            {-1, -1, -1, -1}
        },
        { // Dynamic shapes instances
            {{1, 16, 1, 1}},
            {{8, 1, 1, 1}}
        }
    }
};

INSTANTIATE_TEST_CASE_P(smoke_DynamicShape4D, BroadcastLayerCPUTest,
                    ::testing::Combine(::testing::Combine(
                            ::testing::ValuesIn(dynamicInputShapes4D),
                            ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{8, 16,  1, 7}, {8, 16, 10, 7}}),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::Values(ov::op::BroadcastType::NUMPY),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<bool>>{{true, true}, {false, true}}),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                    BroadcastLayerCPUTest::getTestCaseName);

// 5D
const std::vector<inputShapesPair> staticInputShapes5D = {
    {
        {},
        { // Static shapes
            {{1, 16, 1, 1, 1}}
        }
    }
};
const std::vector<inputShapesPair> dynamicInputShapes5D = {
    {
        { // Origin dynamic shapes
            {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)}
        },
        { // Dynamic shapes instances
            {{1, 16, 1, 1, 1}},
            {{8, 1, 1, 7, 1}},
            {{8, 1, 1, 1, 1}}
        }
    },
    {
        { // Origin dynamic shapes
            {-1, -1, -1, -1, -1}
        },
        { // Dynamic shapes instances
            {{1, 16, 1, 1, 1}},
            {{8, 16, 1, 7, 1}}
        }
    }
};
std::vector<std::vector<int64_t>> targetShapes5D {
    {8, 16,  1, 7, 1},
    {8, 16, 10, 7, 4}
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};

INSTANTIATE_TEST_CASE_P(smoke_StaticShape5D, BroadcastLayerCPUTest,
                    ::testing::Combine(
                        ::testing::Combine(
                            ::testing::ValuesIn(staticInputShapes5D),
                            ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 16, 1, 1, 3}, {1, 16, 3, 1, 3}}),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::ValuesIn({ov::op::BroadcastType::NUMPY}),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<bool>{true, true}),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::ValuesIn(CPUParams5D)),
                    BroadcastLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_DynamicShape5D, BroadcastLayerCPUTest,
                    ::testing::Combine(
                        ::testing::Combine(
                            ::testing::ValuesIn(dynamicInputShapes5D),
                            ::testing::ValuesIn(targetShapes5D),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::ValuesIn({ov::op::BroadcastType::NUMPY}),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<bool>>{{true, true}, {false, true}}),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::ValuesIn(std::vector<CPUSpecificParams>{{{}, {}, {}, "ref"}})),
                    BroadcastLayerCPUTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPULayerTestsDefinitions
