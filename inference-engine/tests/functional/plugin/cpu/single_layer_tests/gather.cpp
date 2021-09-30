// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/gather.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

typedef std::tuple<
        inputShapesPair,                   // Input shapes
        int64_t,                           // Axis
        int64_t,                           // Batch dims
        InferenceEngine::Precision,        // Network precision
        std::string,                       // Device name
        CPUSpecificParams                  // CPU specific params
> GatherLayerTestCPUParams;

class GatherLayerTestCPU : public testing::WithParamInterface<GatherLayerTestCPUParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherLayerTestCPUParams> obj) {
        inputShapesPair inputShapes;
        int axis, batchDims;
        Precision netPrecision;
        std::string targetDevice;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, axis, batchDims, netPrecision, targetDevice, cpuParams) = obj.param;

        std::ostringstream result;
        result << "DynShapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        result << "StatShapes=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
        result << "axis=" << axis << "_";
        result << "batchDims=" << batchDims << "_";
        result << "netPrc=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        inputShapesPair inputShapes;
        int64_t batchDims;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, axis, batchDims, netPrecision, targetDevice, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType = std::string("ref_any_") + netPrecision.name();

        targetStaticShapes.reserve(inputShapes.second.size());
        for (const auto& staticShape : inputShapes.second) {
            targetStaticShapes.push_back({staticShape});
        }
        inputDynamicShapes = { inputShapes.first };
        ov::Shape inputDataShape = targetStaticShapes.front().front(), indicesShape = targetStaticShapes.front().back();
        dataSrcRank = inputDataShape.size();

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector functionParams {
            ngraph::builder::makeParams(ngPrc, { {"data", inputDataShape} })[0],
            ngraph::builder::makeParams(ov::element::i32, { {"indices", indicesShape} })[0]
        };
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));
        auto axisNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({}), { axis });
        auto gather = std::make_shared<ov::op::v8::Gather>(paramOuts[0], paramOuts[1], axisNode, batchDims);
        ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(gather) };
        function = std::make_shared<ov::Function>(results, functionParams, "Gather");
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &inputInfo) const override {
        if (inputInfo.name() == "indices") {
            const auto& td = inputInfo.getTensorDesc();
            size_t normAxis = axis < 0 ? axis + dataSrcRank : axis;
            const auto axDim = targetStaticShapes[index][0][normAxis];
            if (axDim == 1) {
                // Random generator cannot generate values in range [0; 0]
                int values[1] = {0};
                return FuncTestUtils::createAndFillBlobWithFloatArray<int32_t>(td, values, 1);
            } else {
                return FuncTestUtils::createAndFillBlob(td, axDim - 1, 0);
            }
        } else {
            return LayerTestsCommon::GenerateInput(inputInfo);
        }
    }

    int64_t axis = 0;
    int64_t dataSrcRank = 0;
};

TEST_P(GatherLayerTestCPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Gather");
}

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

// 1D
const std::vector<inputShapesPair>
    staticInputShapes1D = {
        {{}, {{{4}, {2, 3, 4}}}},
        {{}, {{{4}, {1}}}},
        {{}, {{{4}, {9}}}},
        {{}, {{{5}, {5}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes1D = {
        {{{ngraph::Dimension(4, 6)}, {ngraph::Dimension(1, 10)}}, {{{4}, {1}}, {{4}, {9}}, /*{{5}, {5}}*/}} // TODO: fix this case
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape1D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes1D),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(std::vector<CPUSpecificParams>{{}})),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape1D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes1D),
                    ::testing::Values(0),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(std::vector<CPUSpecificParams>{{}})),
                GatherLayerTestCPU::getTestCaseName);

// 2D
const std::vector<inputShapesPair>
    staticInputShapes2D = {
        {{}, {{{4, 7}, {4, 55}}}},
        {{}, {{{4, 17}, {4, 17}}}},
        {{}, {{{4, 55}, {4, 7}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes2D = {
        {{{4, ngraph::Dimension(3, 99)}, {4, ngraph::Dimension(3, 99)}}, {{{4, 7}, {4, 55}}, {{4, 55}, {4, 7}}, {{4, 17}, {4, 17}}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape2D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes2D),
                    ::testing::Values(1),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1}),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(std::vector<CPUSpecificParams>{{}})),
                GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape2D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes2D),
                    ::testing::Values(1),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1}),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(std::vector<CPUSpecificParams>{{}})),
                GatherLayerTestCPU::getTestCaseName);

// 4D
const std::vector<inputShapesPair>
    staticInputShapes4D = {
        {{}, {{{4, 5, 6, 7}, {2, 5, 1}}}},
        {{}, {{{10, 5, 6, 7}, {2, 5, 2}}}},
        {{}, {{{16, 5, 6, 7}, {3, 5, 3}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes4D = {
        {{{ngraph::Dimension(4, 20), 5, 6, 7}, {ngraph::Dimension(2, 4), 5, ngraph::Dimension(1, 4)}},
        {{{4, 5, 6, 7}, {2, 5, 1}},
         {{10, 5, 6, 7}, {2, 5, 2}},
         {{16, 5, 6, 7}, {3, 5, 3}}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape4D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(staticInputShapes4D),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2, -1}),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(std::vector<CPUSpecificParams>{{}})),
                GatherLayerTestCPU::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape4D, GatherLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes4D),
                    ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2, -1}),
                    ::testing::Values(0),
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(std::vector<CPUSpecificParams>{{}})),
                GatherLayerTestCPU::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
