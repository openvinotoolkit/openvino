// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/transpose.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

// Since the Transpose ngraph operation is converted to the transpose node, we will use it in the transpose test

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,            // Input order
        InferenceEngine::Precision,     // Net precision
        std::vector<size_t>,            // Input shapes
        std::string,                    // Target device name
        std::map<std::string, std::string>, // Additional network configuration
        CPUSpecificParams> TransposeLayerCPUTestParamSet;

class TransposeLayerCPUTest : public testing::WithParamInterface<TransposeLayerCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeLayerCPUTestParamSet> obj) {
        Precision netPrecision;
        std::vector<size_t> inputShape, inputOrder;
        std::string targetDevice;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputOrder, netPrecision, inputShape, targetDevice, additionalConfig, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "inputOrder=" << CommonTestUtils::vec2str(inputOrder) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        Precision netPrecision;
        std::vector<size_t> inputShape, inputOrder;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputOrder, netPrecision, inputShape, targetDevice, additionalConfig, cpuParams) = this->GetParam();
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        inPrc = outPrc = netPrecision; // since the layer does not convert precisions

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType = std::string("unknown_") + inPrc.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto inOrderShape = inputOrder.empty() ? ngraph::Shape({0}) : ngraph::Shape({inputShape.size()});
        const auto inputOrderOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                                             inOrderShape,
                                                                             inputOrder);
        const auto transpose = std::make_shared<ngraph::opset3::Transpose>(paramOuts.at(0), inputOrderOp);
        transpose->get_rt_info() = getCPUInfo();
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(transpose)};
        function = std::make_shared<ngraph::Function>(results, params, "Transpose");
    }
};

TEST_P(TransposeLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Transpose");
}

namespace {
std::map<std::string, std::string> additional_config;

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams {{ncdhw}, {}, {}, {}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        Precision::I8,
        Precision::BF16,
        Precision::FP32
};

const std::vector<InferenceEngine::Precision> netPrecisionsPerChannels = {
        Precision::I8,
        Precision::FP32
};

const std::vector<std::vector<size_t>> inputShapes4D = {
    {2, 32, 10, 20}
};

const std::vector<std::vector<size_t>> inputOrder4D = {
        std::vector<size_t>{0, 1, 2, 3},
        std::vector<size_t>{0, 2, 3, 1},
        std::vector<size_t>{0, 2, 1, 3},
        std::vector<size_t>{1, 0, 2, 3},
        std::vector<size_t>{},
};

const std::vector<std::vector<size_t>> inputOrderPerChannels4D = {
        std::vector<size_t>{0, 1, 2, 3},
        std::vector<size_t>{0, 2, 1, 3},
        std::vector<size_t>{1, 0, 2, 3},
        std::vector<size_t>{},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nchw,
};

const auto params4D = ::testing::Combine(
        ::testing::ValuesIn(inputOrder4D),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes4D),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::ValuesIn(CPUParams4D));

INSTANTIATE_TEST_CASE_P(smoke_Transpose4D_CPU, TransposeLayerCPUTest, params4D, TransposeLayerCPUTest::getTestCaseName);

const auto paramsPerChannels4D = ::testing::Combine(
        ::testing::ValuesIn(inputOrderPerChannels4D),
        ::testing::ValuesIn(netPrecisionsPerChannels),
        ::testing::ValuesIn(inputShapes4D),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(cpuParams_nhwc));

INSTANTIATE_TEST_CASE_P(smoke_PermutePerChannels4D_CPU, TransposeLayerCPUTest, paramsPerChannels4D, TransposeLayerCPUTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes5D = {
        {2, 32, 5, 10, 20}
};

const std::vector<std::vector<size_t>> inputOrder5D = {
        std::vector<size_t>{0, 1, 2, 3, 4},
        std::vector<size_t>{0, 4, 2, 3, 1},
        std::vector<size_t>{0, 4, 2, 1, 3},
        std::vector<size_t>{0, 2, 3, 4, 1},
        std::vector<size_t>{0, 2, 4, 3, 1},
        std::vector<size_t>{0, 3, 2, 4, 1},
        std::vector<size_t>{0, 3, 1, 4, 2},
        std::vector<size_t>{1, 0, 2, 3, 4},
        std::vector<size_t>{},
};

const std::vector<std::vector<size_t>> inputOrderPerChannels5D = {
        std::vector<size_t>{0, 1, 2, 3, 4},
        std::vector<size_t>{0, 4, 2, 3, 1},
        std::vector<size_t>{0, 4, 2, 1, 3},
        std::vector<size_t>{0, 2, 4, 3, 1},
        std::vector<size_t>{0, 3, 2, 4, 1},
        std::vector<size_t>{0, 3, 1, 4, 2},
        std::vector<size_t>{1, 0, 2, 3, 4},
        std::vector<size_t>{},
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ncdhw,
};

const auto params5D = ::testing::Combine(
        ::testing::ValuesIn(inputOrder5D),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes5D),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::ValuesIn(CPUParams5D));

INSTANTIATE_TEST_CASE_P(smoke_Transpose5D_CPU, TransposeLayerCPUTest, params5D, TransposeLayerCPUTest::getTestCaseName);

const auto paramsPerChannels5D = ::testing::Combine(
        ::testing::ValuesIn(inputOrderPerChannels5D),
        ::testing::ValuesIn(netPrecisionsPerChannels),
        ::testing::ValuesIn(inputShapes5D),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(cpuParams_ndhwc));

INSTANTIATE_TEST_CASE_P(smoke_PermutePerChannels5D_CPU, TransposeLayerCPUTest, paramsPerChannels5D, TransposeLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
