/// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<int> pooledVector;
    std::string mode;
    std::vector<size_t> inputShape;
}  // namespace

typedef std::tuple<
    std::vector<int>,                                     // pooled vector
    std::string,                                          // adapooling mode
    std::vector<size_t>                                   // feature map shape
> AdaPoolSpecificParams;

typedef std::tuple<
        AdaPoolSpecificParams,
        InferenceEngine::Precision,     // Net precision
        LayerTestsUtils::TargetDevice   // Device name
> AdaPoolLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::AdaPoolLayerTestParams,
        CPUSpecificParams> AdaPoolLayerCPUTestParamsSet;

class AdaPoolLayerCPUTest : public testing::WithParamInterface<AdaPoolLayerCPUTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AdaPoolLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        Precision netPr;
        AdaPoolSpecificParams adaPar;
        std::tie(adaPar, netPr, td) = basicParamsSet;
        std::tie(pooledVector, mode, inputShape) = adaPar;
        std::ostringstream result;
        result << "AdaPoolTest_";
        result << std::to_string(obj.index) << "_";
        result << "Ch=" << std::to_string(inputShape[1]) << "_";
        // TODO: ...
//        result << "pooledH=" << pooledH << "_";
//        result << "pooledW=" << pooledW << "_";
//        result << "spatialScale=" << spatialScale << "_";
//        result << "samplingRatio=" << samplingRatio << "_";
        result << (netPr == Precision::FP32 ? "FP32" : "BF16") << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::AdaPoolSpecificParams adaPoolParams;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(adaPoolParams, netPrecision, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        std::tie(pooledVector, mode, inputShape) = adaPoolParams;

        ngraph::Shape coordsShape = { pooledVector.size() };
        auto pooledParam = ngraph::builder::makeConstant<int32_t>(ngraph::element::i32, coordsShape, pooledVector);
        auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});

        // TODO: if
        auto adapool = std::make_shared<ngraph::opset8::AdaptiveMaxPool>(params[0], pooledParam, ngraph::element::i32);
//        (mode == "max" ? std::make_shared<ngraph::opset8::AdaptiveMaxPool>(params[0], pooledParam, ngraph::element::i32) :
//                         std::make_shared<ngraph::opset8::AdaptiveAvgPool>(params[0], pooledParam));
        adapool->set_friendly_name("AdaPool");
        adapool->get_rt_info() = getCPUInfo();
        selectedType = std::string("unknown_") + inPrc.name();

        threshold = 1e-2;
        function = std::make_shared<ngraph::Function>(adapool->outputs(), params, "AdaPool");
    }
};

TEST_P(AdaPoolLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
//    if (mode == "avg") {
//        CheckPluginRelatedResults(executableNetwork, "AdaptiveAvgPooling");
//    } else {
//        CheckPluginRelatedResults(executableNetwork, "AdaptiveMaxPooling");
//    }
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::string dims = "3D") {
    std::vector<CPUSpecificParams> resCPUParams;
    if (mode == "max") {
        if (dims == "5D") {
            resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc, ncdhw}, {}, {}});
        } else if (dims == "4D") {
            resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nchw}, {}, {}});
        } else {
            resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc, ncw}, {}, {}});
        }
    } else {
        if (dims == "5D") {
            resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc}, {}, {}});
        } else if (dims == "4D") {
            resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc}, {}, {}});
        } else {
            resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc}, {}, {}});
        }
    }

    // TODO: if
//    resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {}, {}});
//    if (with_cpu_x86_avx512f()) {
//        resCPUParams.push_back(CPUSpecificParams{{nChw16c, nc, x}, {nChw16c}, {}, {}});
//    } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
//        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc, x}, {nChw8c}, {}, {}});
//    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
//        InferenceEngine::Precision::BF16
};

const std::vector<std::vector<int>> pooled3DVector = {
        { 1 },
        { 3 },
        { 5 }
};
// TODO: names of dimensions
const std::vector<std::vector<int>> pooled4DVector = {
        { 1, 1 },
        { 3, 5 },
        { 5, 5 }
};

const std::vector<std::vector<int>> pooled5DVector = {
        { 1, 1, 1 },
        { 3, 5, 1 },
        { 3, 5, 3 },
};

const std::vector<std::string> modeVector = {
//        "avg",
        "max"
};

const std::vector<std::vector<size_t>> input3DShapeVector = {
        SizeVector({ 1, 2, 1 }),
        SizeVector({ 1, 1, 7 }),
        SizeVector({ 1, 17, 3 }),
        SizeVector({ 3, 17, 5 }),
};

const std::vector<std::vector<size_t>> input4DShapeVector = {
        SizeVector({ 1, 1, 1, 1 }),
        SizeVector({ 1, 3, 1, 1 }),
        SizeVector({ 3, 17, 5, 2 }),
};

const std::vector<std::vector<size_t>> input5DShapeVector = {
        SizeVector({ 1, 1, 1, 1, 1 }),
        SizeVector({ 1, 17, 2, 5, 2 }),
        SizeVector({ 3, 17, 4, 5, 4 }),
};

const auto adaPool3DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(modeVector),            // pooling mode
        ::testing::ValuesIn(input3DShapeVector)     // feature map shape
);

const auto adaPool4DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled4DVector),         // output spatial shape
        ::testing::ValuesIn(modeVector),            // pooling mode
        ::testing::ValuesIn(input4DShapeVector)     // feature map shape
);

const auto adaPool5DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled5DVector),         // output spatial shape
        ::testing::ValuesIn(modeVector),            // pooling mode
        ::testing::ValuesIn(input5DShapeVector)     // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool3DParams,
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("3D"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool4DParams,
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("4D"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool5DParams,
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("5D"))),
                         AdaPoolLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
