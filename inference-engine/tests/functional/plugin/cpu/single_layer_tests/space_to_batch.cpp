// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/space_to_batch.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
        LayerTestsDefinitions::spaceToBatchParamsTuple,
        CPUSpecificParams> SpaceToBatchLayerCPUTestParamSet;

class SpaceToBatchCPULayerTest : public testing::WithParamInterface<SpaceToBatchLayerCPUTestParamSet>,
                                 virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToBatchLayerCPUTestParamSet> &obj) {
        LayerTestsDefinitions::spaceToBatchParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::SpaceToBatchLayerTest::getTestCaseName(
                testing::TestParamInfo<LayerTestsDefinitions::spaceToBatchParamsTuple>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::spaceToBatchParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<size_t> inputShape;
        std::vector<int64_t> blockShape, padsBegin, padsEnd;
        InferenceEngine::Precision netPrecision;
        std::tie(blockShape, padsBegin, padsEnd, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = basicParamsSet;

        inPrc = outPrc = netPrecision;
        selectedType = std::string("unknown_") + netPrecision.name();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto s2b = ngraph::builder::makeSpaceToBatch(paramOuts[0], ngPrc, blockShape, padsBegin, padsEnd);
        s2b->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2b)};
        function = std::make_shared<ngraph::Function>(results, params, "SpaceToBatch");
    }
};

TEST_P(SpaceToBatchCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "SpaceToBatch");
};

namespace {

const std::vector<Precision> precisions = {
        Precision::I8,
        Precision::FP32,
        Precision::BF16
};

const std::vector<std::vector<int64_t>> blockShape4D = {{1, 1, 1, 1}, {1, 1, 1, 2}, {1, 1, 2, 1}, {1, 1, 2, 2}};

const std::vector<std::vector<int64_t>> cropsBegin4D = {{0, 0, 0, 1}, {0, 0, 2, 1}, {0, 0, 4, 1}, {0, 0, 4, 3}};

const std::vector<std::vector<int64_t>> cropsEnd4D   = {{0, 0, 0, 1}, {0, 0, 2, 1}, {0, 0, 4, 1}, {0, 0, 2, 3}};

const std::vector<std::vector<size_t>> inputShapes4D = {{1, 3, 8, 8}, {1, 3, 4, 2}, {1, 6, 4, 4}, {2, 6, 2, 4}};

const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const auto spaceToBatchParamsSet4D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(blockShape4D),
                ::testing::ValuesIn(cropsBegin4D),
                ::testing::ValuesIn(cropsEnd4D),
                ::testing::ValuesIn(inputShapes4D),
                ::testing::ValuesIn(precisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(cpuParams_4D));

INSTANTIATE_TEST_CASE_P(smoke_CPUBatchToSpace_4D, SpaceToBatchCPULayerTest, spaceToBatchParamsSet4D, SpaceToBatchCPULayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> blockShape5D = {{1, 1, 4, 3, 1}, {1, 1, 2, 1, 3}, {1, 1, 4, 1, 1}, {1, 1, 2, 3, 3}};

const std::vector<std::vector<int64_t>> cropsBegin5D = {{0, 0, 1, 0, 0}, {0, 0, 1, 0, 3}, {0, 0, 5, 3, 3}, {0, 0, 5, 6, 3}};

const std::vector<std::vector<int64_t>> cropsEnd5D   = {{0, 0, 1, 0, 0}, {0, 0, 1, 0, 3}, {0, 0, 5, 3, 3}, {0, 0, 5, 6, 3}};

const std::vector<std::vector<size_t>> inputShapes5D = {{2, 6, 10, 3, 3}, {3, 8, 30, 6, 9}, {5, 2, 10, 12, 12}, {7, 7, 30, 12, 12}};

const std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const auto spaceToBatchParamsSet5D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(blockShape5D),
                ::testing::ValuesIn(cropsBegin5D),
                ::testing::ValuesIn(cropsEnd5D),
                ::testing::ValuesIn(inputShapes5D),
                ::testing::ValuesIn(precisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(cpuParams_5D));

INSTANTIATE_TEST_CASE_P(smoke_BatchToSpaceLayerTest_5D, SpaceToBatchCPULayerTest, spaceToBatchParamsSet5D, SpaceToBatchCPULayerTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions