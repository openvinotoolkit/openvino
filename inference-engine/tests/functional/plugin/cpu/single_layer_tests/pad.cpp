// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/pad.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::padLayerTestParamsSet,
        CPUSpecificParams
> padLayerCPUTestParamsSet;

class PadLayerCPUTest : public testing::WithParamInterface<padLayerCPUTestParamsSet>,
                        public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<padLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::padLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::PadLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::padLayerTestParamsSet>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::padLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::SizeVector inputShape;
        std::vector<int64_t> padsBegin, padsEnd;
        float argPadValue;
        ngraph::helpers::PadMode padMode;
        InferenceEngine::Precision netPrecision;
        std::tie(padsBegin, padsEnd, argPadValue, padMode, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
                basicParamsSet;

        inPrc = outPrc = netPrecision;
        selectedType = std::string("ref_") + netPrecision.name();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
        auto pad = ngraph::builder::makePad(paramOuts[0], padsBegin, padsEnd, argPadValue, padMode);
        pad->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pad)};
        function = std::make_shared<ngraph::Function>(results, params, "pad");
    }
};

TEST_P(PadLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Pad");
}

namespace {


const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, {}};


const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const std::vector<float> argPadValue = {0.f, 1.f, 2.5f, -1.f};

const std::vector<ngraph::helpers::PadMode> padMode = {
        ngraph::helpers::PadMode::EDGE,
        ngraph::helpers::PadMode::REFLECT,
        ngraph::helpers::PadMode::SYMMETRIC
};

const std::vector<std::vector<int64_t>> padsBegin4DConstBlocked = {{0, 0, 0, 0}, {0, 0, 1, 3}, {2, 16, 1, 0}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DConstBlocked   = {{0, 0, 0, 0}, {0, 0, 2, 1}, {2, 0, 0, 1}, {1, 32, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4DBlocked = {{0, 0, 0, 0}, {0, 0, 1, 3}, {2, 0, 1, 0}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DBlocked   = {{0, 0, 0, 0}, {0, 0, 2, 1}, {2, 0, 0, 1}, {1, 0, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 0, 0, 0}, {0, 1, 1, 1}, {0, 2, 1, 0}, {0, 0, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D   = {{0, 0, 0, 0}, {0, 2, 1, 1}, {0, 0, 2, 0}, {1, 1, 0, 0}};

const std::vector<CPUSpecificParams> CPUParams4DBlocked = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
};

const auto pad4DConstParamsBlocked = testing::Combine(
        testing::ValuesIn(padsBegin4DConstBlocked),
        testing::ValuesIn(padsEnd4DConstBlocked),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 5, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                pad4DConstParamsBlocked,
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);


const auto pad4DConstParams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 5, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                pad4DConstParams,
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

const auto pad4DParamsBlocked = testing::Combine(
        testing::ValuesIn(padsBegin4DBlocked),
        testing::ValuesIn(padsEnd4DBlocked),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 10, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                pad4DParamsBlocked,
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

const auto pad4DParams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 10, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4D,
        PadLayerCPUTest,
        ::testing::Combine(
                pad4DParams,
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

const std::vector<std::vector<int64_t>> padsBegin5DConstBlocked = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 0}, {2, 32, 1, 1, 0}, {0, 0, 1, 3, 1}, {0, 0, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DConstBlocked   = {{0, 0, 0, 0, 0}, {1, 16, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}, {0, 0, 1, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin5DBlocked = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 0}, {2, 0, 1, 1, 0}, {0, 0, 1, 3, 1}, {0, 0, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DBlocked   = {{0, 0, 0, 0, 0}, {1, 0, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}, {0, 0, 1, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin5D = {{0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {1, 1, 1, 1, 0}, {2, 0, 1, 0, 1}, {0, 2, 1, 3, 1}};
const std::vector<std::vector<int64_t>> padsEnd5D   = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {1, 0, 1, 1, 2}, {2, 2, 0, 1, 0}, {1, 1, 2, 0, 1}};

const std::vector<CPUSpecificParams> CPUParams5DBlocked = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
};

const auto pad5DConstParamsBlocked = testing::Combine(
        testing::ValuesIn(padsBegin5DConstBlocked),
        testing::ValuesIn(padsEnd5DConstBlocked),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 5, 5, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                pad5DConstParamsBlocked,
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

const auto pad5DConstParams = testing::Combine(
        testing::ValuesIn(padsBegin5D),
        testing::ValuesIn(padsEnd5D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 10, 5, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                pad5DConstParams,
                ::testing::Values(cpuParams_ndhwc)),
        PadLayerCPUTest::getTestCaseName
);

const auto pad5DParamsBlocked = testing::Combine(
        testing::ValuesIn(padsBegin5DBlocked),
        testing::ValuesIn(padsEnd5DBlocked),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 5, 5, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                pad5DParamsBlocked,
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

const auto pad5DParams = testing::Combine(
        testing::ValuesIn(padsBegin5D),
        testing::ValuesIn(padsEnd5D),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::ValuesIn(inputPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 16, 5, 5, 5}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5D,
        PadLayerCPUTest,
        ::testing::Combine(
                pad5DParams,
                ::testing::Values(cpuParams_ndhwc)),
        PadLayerCPUTest::getTestCaseName
);


} // namespace
} // namespace CPULayerTestsDefinitions

