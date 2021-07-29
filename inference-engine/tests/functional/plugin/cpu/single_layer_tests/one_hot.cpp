// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_functions/builders.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,       // Input shape
        int,                       // axis to extend
        size_t,                    // depth
        float,                     // on_value
        float,                     // off_value
        InferenceEngine::Precision,// Net precision
        InferenceEngine::Precision,// Input precision
        InferenceEngine::Precision,// Output precision
        std::string,               // Target device name
        CPUSpecificParams
> oneHotCPUTestParams;

class OneHotLayerCPUTest : public testing::WithParamInterface<oneHotCPUTestParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHotCPUTestParams>& obj) {
        InferenceEngine::SizeVector inputShape;
        int axis;
        size_t depth;
        float onValue, offValue;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        std::string targetDevice;
        CPUSpecificParams cpuParams;
        std::tie(inputShape, axis, depth, onValue, offValue, netPrecision, inPrc, outPrc, targetDevice, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "axis=" << axis << "_";
        result << "depth=" << depth << "_";
        result << "OnVal=" << onValue << "_";
        result << "OffVal=" << offValue << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        ngraph::Shape inputShape;
        int axis;
        size_t depth;
        float onValue, offValue;
        InferenceEngine::Precision netPrecision;
        CPUSpecificParams cpuParams;

        std::tie(inputShape, axis, depth, onValue, offValue, netPrecision, inPrc, outPrc, targetDevice, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = std::string("ref_any_") + inPrc.name();

        auto ngOutPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
        auto depthConst = ngraph::builder::makeConstant<size_t>(ngraph::element::i32, {}, {depth});
        auto onConst = ngraph::builder::makeConstant<float>(ngOutPrc, {}, {onValue});
        auto offConst = ngraph::builder::makeConstant<float>(ngOutPrc, {}, {offValue});

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto inputParams = ngraph::builder::makeParams(ngPrc, { inputShape });

        auto oneHot = std::make_shared<ngraph::opset5::OneHot>(inputParams.front(), depthConst, onConst, offConst, axis);
        function = makeNgraphFunction(ngPrc, inputParams, oneHot, "OneHot");
    }
};

TEST_P(OneHotLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "OneHot");
}

namespace {
const std::vector<Precision> inPrc = {Precision::I32};
const std::vector<Precision> outPrc = {Precision::FP32, Precision::BF16, Precision::I8, Precision::U8};

// 0d -> 1d, depth
const auto testCase_1d = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{}),
        ::testing::Values(-1, 0),
        ::testing::Values(3, 4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(inPrc),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_1D, OneHotLayerCPUTest, testCase_1d, OneHotLayerCPUTest::getTestCaseName);


// 1d -> 2d, axis default
const auto testCase_2d = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{3}),
        ::testing::Values(-1, 0, 1),
        ::testing::Values(6),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(inPrc),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_2D, OneHotLayerCPUTest, testCase_2d, OneHotLayerCPUTest::getTestCaseName);

// 2d -> 3d, on_value, off_value
const auto testCase_3d = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{3, 2}),
        ::testing::Values(-1, 0, 1),
        ::testing::Values(4),
        ::testing::Values(2.f),
        ::testing::Values(-1.f),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(inPrc),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_3D, OneHotLayerCPUTest, testCase_3d, OneHotLayerCPUTest::getTestCaseName);

// 3d -> 4d
const auto testCase_4d = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 3, 2}),
        ::testing::Values(-1, 0, 1, 2),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(inPrc),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_4D, OneHotLayerCPUTest, testCase_4d, OneHotLayerCPUTest::getTestCaseName);

// 4d -> 5d
const auto testCase_5d = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 3, 2, 3}),
        ::testing::Values(-1, 0, 1, 2, 3),
        ::testing::Values(4),
        ::testing::Values(1.f),
        ::testing::Values(0.f),
        ::testing::Values(Precision::I32),
        ::testing::ValuesIn(inPrc),
        ::testing::ValuesIn(outPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(emptyCPUSpec)
);
INSTANTIATE_TEST_SUITE_P(smoke_OneHotCPU_5D, OneHotLayerCPUTest, testCase_5d, OneHotLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions