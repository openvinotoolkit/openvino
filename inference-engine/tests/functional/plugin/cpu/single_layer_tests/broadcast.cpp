// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/broadcast.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using BroadcastLayerTestParamsSet = LayerTestsDefinitions::BroadcastParamsTuple;

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

        std::ostringstream result;
        result << LayerTestsDefinitions::BroadcastLayerTest::getTestCaseName(testing::TestParamInfo<BroadcastLayerTestParamsSet>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::SizeVector targetShape;
        ngraph::AxisSet axesMapping;
        ngraph::op::BroadcastType mode;
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision networkPrecision;
        std::tie(targetShape, axesMapping, mode, inputShape, networkPrecision, targetDevice) = basicParamsSet;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(networkPrecision);
        auto target_shape_const = ngraph::opset3::Constant::create(ngraph::element::i64, {targetShape.size()}, targetShape);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto broadcast = std::dynamic_pointer_cast<ngraph::opset3::Broadcast>(
                ngraph::builder::makeBroadcast(params[0], target_shape_const, mode, axesMapping));
        broadcast->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(broadcast)};
        function = std::make_shared<ngraph::Function>(results, params, "broadcast");
    }
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
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::BOOL
};
/* ============= */

/* INSTANCES */
// 4D
const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc,
};

const auto numpyBroadcastParams4D = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 16, 1, 3},
                          std::vector<size_t>{1, 16, 3, 3}),
        ::testing::Values(ngraph::AxisSet{}),
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>({1, 16, 1, 1})),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));


INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast4D, BroadcastLayerCPUTest,
                        ::testing::Combine(numpyBroadcastParams4D, ::testing::ValuesIn(CPUParams4D)), BroadcastLayerCPUTest::getTestCaseName);

// 5D
const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};

const auto numpyBroadcastParams5D = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 16, 1, 1, 3},
                          std::vector<size_t>{1, 16, 3, 1, 3}),
        ::testing::Values(ngraph::AxisSet{}),
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>({1, 16, 1, 1, 1})),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast5D, BroadcastLayerCPUTest,
                        ::testing::Combine(numpyBroadcastParams5D, ::testing::ValuesIn(CPUParams5D)), BroadcastLayerCPUTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPULayerTestsDefinitions
