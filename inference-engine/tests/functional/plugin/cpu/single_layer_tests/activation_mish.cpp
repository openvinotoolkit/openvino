// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/activation_mish.hpp>
#include "cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<cpu_memory_format_t>,
        std::vector<cpu_memory_format_t>,
        std::vector<std::string>,
        std::string> mishCPUSpecificParams;

typedef std::tuple<
        mishLayerTestParamsSet,
        mishCPUSpecificParams> mishLayerCPUTestParamsSet;

class MishLayerCPUTest : public testing::WithParamInterface<mishLayerCPUTestParamsSet>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<mishLayerCPUTestParamsSet> obj) {
        mishLayerTestParamsSet basicParamsSet;
        mishCPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::MishLayerTest::getTestCaseName(testing::TestParamInfo<mishLayerTestParamsSet>(
                basicParamsSet, 0));

        std::vector<cpu_memory_format_t> inFmts, outFmts;
        std::vector<std::string> priority;
        std::string selectedType;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        result << "_inFmts=" << CPUTestUtils::fmts2str(inFmts);
        result << "_outFmts=" << CPUTestUtils::fmts2str(outFmts);
        result << "_primitive=" << selectedType;

        return result.str();
    }

protected:
    void SetUp() {
        mishLayerTestParamsSet basicParamsSet;
        mishCPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<size_t> inputShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(netPrecision, inputShape, targetDevice) = basicParamsSet;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto mish = std::dynamic_pointer_cast<ngraph::op::v4::Mish>(
                ngraph::builder::makeActivationMish(paramOuts[0]));
        mish->get_rt_info() = CPUTestUtils::setCPUInfo(inFmts, outFmts, priority);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mish)};
        function = std::make_shared<ngraph::Function>(results, params, "mish");
    }

    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};

TEST_P(MishLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CPUTestUtils::CheckCPUImpl(executableNetwork, "Mish", inFmts, outFmts, selectedType);
}

namespace {

/* CPU PARAMS */
const auto cpuParams_ref_2D = mishCPUSpecificParams{{nchw}, {nchw}, {"ref_any"}, "ref_any_FP32"};
const auto cpuParams_ref_3D = mishCPUSpecificParams{{ncdhw}, {ncdhw}, {"ref_any"}, "ref_any_FP32"};

const auto cpuParams_gemm_2D = mishCPUSpecificParams{{nchw}, {nchw}, {"gemm_any"}, "jit_gemm_FP32"};
const auto cpuParams_gemm_3D = mishCPUSpecificParams{{ncdhw}, {ncdhw}, {"gemm_any"}, "jit_gemm_FP32"};

const auto cpuParams_sse42_2D = mishCPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"};
const auto cpuParams_sse42_3D = mishCPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42_FP32"};

const auto cpuParams_avx2_2D = mishCPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"};
const auto cpuParams_avx2_3D = mishCPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2_FP32"};

const auto cpuParams_avx512_2D = mishCPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"};
const auto cpuParams_avx512_3D = mishCPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512_FP32"};
/* ========== */

/* MISH TEST UTILS */
std::vector<mishCPUSpecificParams> filterCPUInfoForDevice(std::vector<mishCPUSpecificParams> CPUParams) {
    std::vector<mishCPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("jit") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !with_cpu_x86_avx512f())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

std::vector<mishLayerCPUTestParamsSet> filterParamsSetForDevice(std::vector<mishLayerCPUTestParamsSet> paramsSet) {
    std::vector<mishLayerCPUTestParamsSet> resParamsSet;
    const int cpuParamsIndex = 1;
    const int selectedTypeIndex = 3;

    for (auto param : paramsSet) {
        auto cpuParams = std::get<cpuParamsIndex>(param);
        auto selectedTypeStr = std::get<selectedTypeIndex>(cpuParams);

        if (selectedTypeStr.find("jit") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !with_cpu_x86_avx512f())
            continue;

        resParamsSet.push_back(param);
    }

    return resParamsSet;
}
/* ===================== */

/* INSTANCES */
/* ============= Mish (Planar 2D) ============= */
const std::vector<mishCPUSpecificParams> CPUParams_Planar_2D = {
        cpuParams_gemm_2D
};

INSTANTIATE_TEST_CASE_P(Mish_2D_Planar_FP32, MishLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(std::vector<size_t >({2, 12, 7, 7})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_2D))),
                        MishLayerCPUTest::getTestCaseName);

/* ============= Mish (Planar 3D) ============= */
const std::vector<mishCPUSpecificParams> CPUParams_Planar_3D = {
        cpuParams_gemm_3D
};

INSTANTIATE_TEST_CASE_P(Mish_3D_Planar_FP32, MishLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(std::vector<size_t >({2, 12, 7, 7, 7})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_3D))),
                        MishLayerCPUTest::getTestCaseName);

/* ============= Mish (Blocked 2D) ============= */
const std::vector<mishCPUSpecificParams> CPUParams_Blocked_2D = {
        cpuParams_sse42_2D,
        cpuParams_avx2_2D,
        cpuParams_avx512_2D
};

INSTANTIATE_TEST_CASE_P(Mish_Blocked_FP32, MishLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(std::vector<size_t >({2, 64, 7, 7})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_2D))),
                        MishLayerCPUTest::getTestCaseName);

/* ============= Mish (Blocked 3D) ============= */
const std::vector<mishCPUSpecificParams> CPUParams_Blocked_3D = {
        cpuParams_avx2_3D,
        cpuParams_avx512_3D
};

INSTANTIATE_TEST_CASE_P(Mish_3D_Blocked_FP32, MishLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(std::vector<size_t >({2, 64, 7, 7, 7})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_3D))),
                        MishLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
