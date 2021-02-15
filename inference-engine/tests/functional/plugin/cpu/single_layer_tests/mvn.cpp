// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/mvn.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::mvnParams,
        CPUSpecificParams,
        Precision, // CNNNetwork input precision
        Precision> // CNNNetwork output precision
MvnLayerCPUTestParamSet;

class MvnLayerCPUTest : public testing::WithParamInterface<MvnLayerCPUTestParamSet>,
                        virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MvnLayerCPUTestParamSet> obj) {
        LayerTestsDefinitions::mvnParams basicParamsSet;
        CPUSpecificParams cpuParams;
        Precision inputPrecision, outputPrecision;
        std::tie(basicParamsSet, cpuParams, inputPrecision, outputPrecision) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::MvnLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::mvnParams>(
                basicParamsSet, 0));

        result << "_" << "CNNInpPrc=" << inputPrecision.name();
        result << "_" << "CNNOutPrc=" << outputPrecision.name();

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        LayerTestsDefinitions::mvnParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams, inPrc, outPrc) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision netPrecision;
        bool acrossChanels, normalizeVariance;
        double eps;
        std::tie(inputShapes, netPrecision, acrossChanels, normalizeVariance, eps, targetDevice) = basicParamsSet;
        auto netPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto param = ngraph::builder::makeParams(netPrc, {inputShapes});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
        auto mvn = ngraph::builder::makeMVN(paramOuts[0], acrossChanels, normalizeVariance, eps);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};

        selectedType = getPrimitiveType() + "_" + inPrc.name();

        threshold = 0.015f;

        mvn->get_rt_info() = getCPUInfo();

        function = std::make_shared<ngraph::Function>(results, param, "mvn");
    }
};

TEST_P(MvnLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "MVN");
}

namespace {
const std::vector<std::vector<size_t>> inputShapes_3D = {
        {1, 32, 17},
        {1, 37, 9},
};

const std::vector<std::vector<size_t>> inputShapes_4D = {
        {1, 16, 5, 8},
        {2, 19, 5, 10},
        {7, 32, 2, 8},
        {5, 8, 3, 5},
        {1, 2, 7, 5},
        {1, 4, 5, 5},
        {1, 7, 3, 5},
        {1, 15, 9, 5},
        {4, 41, 6, 9}
};

const std::vector<std::vector<size_t>> inputShapes_5D = {
        {1, 32, 8, 1, 6},
        {1, 9, 1, 15, 9},
        {6, 64, 6, 1, 18},
        {2, 31, 2, 9, 1},
        {10, 16, 5, 10, 6}
};

const std::vector<bool> acrossChannels = {
        true,
        false
};

const std::vector<bool> normalizeVariance = {
        true,
        false
};

const std::vector<double> epsilon = {
        0.000000001
};

std::vector<Precision> inpOutPrc = {Precision::BF16, Precision::FP32};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const auto Mvn3D = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inputShapes_3D),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::ValuesIn(acrossChannels),
            ::testing::ValuesIn(normalizeVariance),
            ::testing::ValuesIn(epsilon),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(emptyCPUSpec),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_3D, MvnLayerCPUTest, Mvn3D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inputShapes_4D),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::ValuesIn(acrossChannels),
                ::testing::ValuesIn(normalizeVariance),
                ::testing::ValuesIn(epsilon),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_4D, MvnLayerCPUTest, Mvn4D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inputShapes_5D),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::ValuesIn(acrossChannels),
                ::testing::ValuesIn(normalizeVariance),
                ::testing::ValuesIn(epsilon),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_5D, MvnLayerCPUTest, Mvn5D, MvnLayerCPUTest::getTestCaseName);


} // namespace
} // namespace CPULayerTestsDefinitions