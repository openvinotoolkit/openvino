// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using TileLayerTestParamsSet = typename std::tuple<
        inputShapesPair,                       // Input shapes
        std::vector<int64_t>,                  // Repeats
        InferenceEngine::Precision,            // Network precision
        bool,                                  // Is Repeats input constant
        std::string>;                          // Device name

typedef std::tuple<
        TileLayerTestParamsSet,
        CPUSpecificParams> TileLayerCPUTestParamsSet;

class TileLayerCPUTest : public testing::WithParamInterface<TileLayerCPUTestParamsSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TileLayerCPUTestParamsSet> obj) {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        inputShapesPair inputShapes;
        std::vector<int64_t> repeats;
        InferenceEngine::Precision netPrecision;
        bool isRepeatsConst;
        std::string deviceName;
        std::tie(inputShapes, repeats, netPrecision, isRepeatsConst, deviceName) = basicParamsSet;

        std::ostringstream result;
        result << "DynShapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        result << "StatShapes=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
        result << "Repeats=" << CommonTestUtils::vec2str(repeats)  << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "constRepeats=" << (isRepeatsConst ? "True" : "False") << "_";
        result << "trgDev=" << deviceName;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        inputShapesPair inputShapes;
        InferenceEngine::Precision netPrecision;
        bool isRepeatsConst;
        std::tie(inputShapes, repeatsData, netPrecision, isRepeatsConst, targetDevice) = basicParamsSet;

        selectedType += std::string("_") + netPrecision.name();

        const size_t dynShapesNum = std::min(inputShapes.first.size(), isRepeatsConst ? 1lu : 2lu);
        for (size_t i = 0lu; i < dynShapesNum; i++) {
            inputDynamicShapes.push_back(inputShapes.first[i]);
            if (!isRepeatsConst) {
                inputDynamicShapes.push_back({ static_cast<int64_t>(repeatsData.size()) });
            }
        }
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            targetStaticShapes.push_back({inputShapes.second[i]});
            if (!isRepeatsConst)
                targetStaticShapes[i].push_back({ repeatsData.size() });
        }

        const auto& inputDataShape = targetStaticShapes.front().front();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector functionParams = ngraph::builder::makeParams(ngPrc, { {"data", inputDataShape} });
        if (!isRepeatsConst) {
            functionParams.push_back(ngraph::builder::makeParams(ov::element::i64, { {"repeats", { repeatsData.size() }} })[0]);
        }

        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));
        std::shared_ptr<ov::Node> tileNode;
        if (isRepeatsConst) {
            tileNode = std::make_shared<ov::op::v0::Tile>(paramOuts[0],
                    ov::op::v0::Constant::create(ov::element::i64, { repeatsData.size() }, repeatsData));
        } else {
            tileNode = std::make_shared<ov::op::v0::Tile>(paramOuts[0], paramOuts[1]);
        }
        tileNode->get_rt_info() = getCPUInfo();
        ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(tileNode) };
        function = std::make_shared<ov::Function>(results, functionParams, "CPUTile");
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &inputInfo) const override {
        if (inputInfo.name() == "repeats") {
            const auto& td = inputInfo.getTensorDesc();
            return FuncTestUtils::createAndFillBlobWithFloatArray<int64_t>(td, repeatsData.data(), repeatsData.size());
        } else {
            return LayerTestsCommon::GenerateInput(inputInfo);
        }
    }

    std::vector<int64_t> repeatsData;
};

TEST_P(TileLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Tile");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, "ref"};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "ref"};

const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* PARAMS */
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I8
};

const std::vector<inputShapesPair> staticInputShapes4D = {
    {
        {},
        { // Static shapes
            {{2, 16, 3, 4}}
        }
    },
    {
        {},
        { // Static shapes
            {{1, 16, 1, 1}}
        }
    }
};
const std::vector<inputShapesPair> dynamicInputShapes4D = {
    {
        { // Origin dynamic shapes
            {ov::Dimension(1, 20), ov::Dimension(10, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)}
        },
        { // Dynamic shapes instances
            {{2, 16, 3, 4}},
            {{1, 16, 1, 1}},
            {{1, 16, 2, 3}}
        }
    },
    {
        { // Origin dynamic shapes
            {-1, -1, -1, -1}
        },
        { // Dynamic shapes instances
            {{3, 15, 5, 7}},
            {{4, 55, 8, 24}}
        }
    }
};

const std::vector<inputShapesPair> staticInputShapes5D = {
    {
        {},
        { // Static shapes
            {{2, 16, 2, 3, 4}}
        }
    }
};
const std::vector<inputShapesPair> dynamicInputShapes5D = {
    {
        { // Origin dynamic shapes
            {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 70)}
        },
        { // Dynamic shapes instances
            {{2, 16, 2, 3, 4}},
            {{1, 16, 8, 5, 4}},
            {{8, 1, 2, 3, 64}}
        }
    },
    {
        { // Origin dynamic shapes
            {-1, -1, -1, -1, -1}
        },
        { // Dynamic shapes instances
            {{2, 16, 2, 3, 4}},
            {{1, 16, 8, 5, 4}},
            {{8, 1, 2, 3, 64}}
        }
    }
};

const std::vector<std::vector<int64_t>> repeats4D = {
        {2, 3},
        {1, 2, 3},
        {1, 1, 1, 1},
        {1, 1, 2, 3},
        {1, 2, 1, 3},
        {2, 1, 1, 1},
        {2, 3, 1, 1},
//        {2, 3, 2, 3, 2},
};
const std::vector<std::vector<int64_t>> repeats5D = {
        {1, 2, 3},
        {1, 1, 2, 3},
        {1, 1, 1, 2, 3},
        {1, 2, 1, 1, 3},
        {2, 1, 1, 1, 1},
        {2, 3, 1, 1, 1},
//        {2, 3, 2, 3, 2, 3},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nchw,
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc,
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_ncdhw,
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};
/* ============= */

/* INSTANCES */
INSTANTIATE_TEST_CASE_P(smoke_StaticShape4D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(staticInputShapes4D),
                                        ::testing::ValuesIn(repeats4D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(true),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams4D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_DynamicShape4D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(dynamicInputShapes4D),
                                        ::testing::ValuesIn(repeats4D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(true, false),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_StaticShape5D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(staticInputShapes5D),
                                        ::testing::ValuesIn(repeats5D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(true),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams5D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_DynamicShape5D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(dynamicInputShapes5D),
                                        ::testing::ValuesIn(repeats5D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(true, false),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        TileLayerCPUTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPULayerTestsDefinitions
