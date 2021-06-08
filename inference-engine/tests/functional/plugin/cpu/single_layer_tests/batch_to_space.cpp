// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/batch_to_space.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
        LayerTestsDefinitions::batchToSpaceParamsTuple,
        CPUSpecificParams> BatchToSpaceLayerCPUTestParamSet;

class BatchToSpaceCPULayerTest : public testing::WithParamInterface<BatchToSpaceLayerCPUTestParamSet>,
                                 virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchToSpaceLayerCPUTestParamSet> &obj) {
        LayerTestsDefinitions::batchToSpaceParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::ostringstream result;
        result << LayerTestsDefinitions::BatchToSpaceLayerTest::getTestCaseName(
                testing::TestParamInfo<LayerTestsDefinitions::batchToSpaceParamsTuple>(basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::batchToSpaceParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::vector<size_t> inputShape;
        std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
        InferenceEngine::Precision netPrecision;
        std::tie(blockShape, cropsBegin, cropsEnd, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;

        if (strcmp(netPrecision.name(), "U8") == 0)
            selectedType = std::string("ref_any_") + "I8";
        else
            selectedType = std::string("ref_any_") + netPrecision.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto b2s = ngraph::builder::makeBatchToSpace(paramOuts[0], ngPrc, blockShape, cropsBegin, cropsEnd);
        b2s->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(b2s)};
        function = std::make_shared<ngraph::Function>(results, params, "BatchToSpace");
    }
};

TEST_P(BatchToSpaceCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "BatchToSpace");
};

namespace {

const std::vector<Precision> precisions = {
        Precision::U8,
        Precision::I8,
        Precision::I32,
        Precision::FP32,
        Precision::BF16
};

const std::vector<std::vector<int64_t>> blockShape4D1  = {{1, 1, 1, 2}, {1, 2, 2, 1}};
const std::vector<std::vector<int64_t>> cropsBegin4D1  = {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> cropsEnd4D1    = {{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 1}};
const std::vector<std::vector<size_t>> inputShapes4D1  = {{8, 16, 10, 10}, {16, 64, 13, 16}};

const std::vector<std::vector<int64_t>> blockShape4D2  = {{1, 2, 3, 4}, {1, 3, 4, 2}};
const std::vector<std::vector<int64_t>> cropsBegin4D2  = {{0, 0, 0, 1}, {0, 0, 1, 2}};
const std::vector<std::vector<int64_t>> cropsEnd4D2    = {{0, 0, 1, 0}, {0, 0, 3, 1}};
const std::vector<std::vector<size_t>> inputShapes4D2  = {{48, 16, 7, 8}, {24, 32, 6, 6}};

const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const auto batchToSpaceParamsSet4D1 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({blockShape4D1})),
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({cropsBegin4D1})),
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({cropsEnd4D1})),
                ::testing::ValuesIn(std::vector<std::vector<size_t>> ({inputShapes4D1})),
                ::testing::ValuesIn(precisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                ::testing::ValuesIn(cpuParams_4D));

const auto batchToSpaceParamsSet4D2 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({blockShape4D2})),
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({cropsBegin4D2})),
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({cropsEnd4D2})),
                ::testing::ValuesIn(std::vector<std::vector<size_t>> ({inputShapes4D2})),
                ::testing::ValuesIn(precisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(cpuParams_4D));

INSTANTIATE_TEST_CASE_P(smoke_BatchToSpaceCPULayerTestCase1_4D, BatchToSpaceCPULayerTest,
                            batchToSpaceParamsSet4D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BatchToSpaceCPULayerTestCase2_4D, BatchToSpaceCPULayerTest,
                            batchToSpaceParamsSet4D2, BatchToSpaceCPULayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> blockShape5D1  = {{1, 1, 2, 2, 1}, {1, 2, 1, 2, 2}};
const std::vector<std::vector<int64_t>> cropsBegin5D1  = {{0, 0, 0, 0, 0}, {0, 0, 0, 3, 3}};
const std::vector<std::vector<int64_t>> cropsEnd5D1    = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 1}};
const  std::vector<std::vector<size_t>> inputShapes5D1 = {{8, 16, 4, 10, 10}, {16, 32, 5, 8, 12}};

const std::vector<std::vector<int64_t>> blockShape5D2  = {{1, 2, 4, 3, 1}, {1, 1, 2, 4, 3}};
const std::vector<std::vector<int64_t>> cropsBegin5D2  = {{0, 0, 1, 2, 0}, {0, 0, 1, 0, 1}};
const std::vector<std::vector<int64_t>> cropsEnd5D2    = {{0, 0, 1, 0, 1}, {0, 0, 1, 1, 1}};
const  std::vector<std::vector<size_t>> inputShapes5D2 = {{48, 16, 3, 3, 3}, {24, 32, 5, 3, 5}};

const std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({nCdhw8c}, {nCdhw8c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const auto batchToSpaceParamsSet5D1 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(blockShape5D1),
                ::testing::ValuesIn(cropsBegin5D1),
                ::testing::ValuesIn(cropsEnd5D1),
                ::testing::ValuesIn(inputShapes5D1),
                ::testing::ValuesIn(precisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)), ::testing::ValuesIn(cpuParams_5D));

const auto batchToSpaceParamsSet5D2 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(blockShape5D2),
                ::testing::ValuesIn(cropsBegin5D2),
                ::testing::ValuesIn(cropsEnd5D2),
                ::testing::ValuesIn(inputShapes5D2),
                ::testing::ValuesIn(precisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)), ::testing::ValuesIn(cpuParams_5D));

INSTANTIATE_TEST_CASE_P(smoke_BatchToSpaceCPULayerTestCase1_5D, BatchToSpaceCPULayerTest,
                            batchToSpaceParamsSet5D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BatchToSpaceCPULayerTestCase2_5D, BatchToSpaceCPULayerTest,
                            batchToSpaceParamsSet5D2, BatchToSpaceCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace CPULayerTestsDefinitions
