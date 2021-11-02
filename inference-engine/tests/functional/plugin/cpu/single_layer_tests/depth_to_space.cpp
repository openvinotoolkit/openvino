// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/depth_to_space.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"


using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;

namespace CPULayerTestsDefinitions {

using DepthToSpaceInputShapes = std::pair<std::vector<ov::PartialShape>, std::vector<ov::Shape>>;
using DepthToSpaceLayerCPUTestParamSet = std::tuple<
        DepthToSpaceInputShapes,                        // Input shape
        InferenceEngine::Precision,                     // Input precision
        DepthToSpace::DepthToSpaceMode,                 // Mode
        std::size_t,                                    // Block size
        CPUSpecificParams
>;

class DepthToSpaceLayerCPUTest : public testing::WithParamInterface<DepthToSpaceLayerCPUTestParamSet>,
                                 virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceLayerCPUTestParamSet> obj) {
        DepthToSpaceInputShapes inputShapes;
        InferenceEngine::Precision inPrc;
        DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, inPrc, mode, blockSize, cpuParams) = obj.param;

        std::ostringstream results;
        if (!inputShapes.first.empty()) {
            results << "IS=(";
            results << CommonTestUtils::partialShape2str(inputShapes.first) << ")_";
        }
        results << "TS=";
        for (const auto& shape : inputShapes.second) {
            results << CommonTestUtils::vec2str(shape) << "_";
        }
        results << "Prc=" << inPrc << "_";
        switch (mode) {
            case DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
                results << "BLOCKS_FIRST_";
                break;
            case DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
                results << "DEPTH_FIRST_";
                break;
            default:
                throw std::runtime_error("Unsupported DepthToSpaceMode");
        }
        results << "BS=" << blockSize << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }
protected:
    void SetUp() override {
        DepthToSpaceInputShapes inputShapes;
        InferenceEngine::Precision inPrc;
        DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, inPrc, mode, blockSize, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = selectedType + "_" + inPrc.name();
        targetDevice = CommonTestUtils::DEVICE_CPU;

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        if (!inputShapes.first.empty()) {
            inputDynamicShapes = inputShapes.first;
        } else {
            inputDynamicShapes = { inputShapes.second.front() };
        }
        for (size_t i = 0; i < inputShapes.second.size(); i++) {
            targetStaticShapes.push_back(std::vector<ngraph::Shape>{inputShapes.second[i]});
        }

        auto params = ngraph::builder::makeDynamicParams(ngPrc, inputDynamicShapes);
        auto d2s = ngraph::builder::makeDepthToSpace(params[0], mode, blockSize);
        d2s->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(d2s)};
        function = std::make_shared<ngraph::Function>(results, params, "DepthToSpaceCPU");
    }
};

TEST_P(DepthToSpaceLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    // TODO: need to uncomment when this method will be updated
    // CheckPluginRelatedResults(executableNetwork, "DepthToSpace");
}

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const std::vector<DepthToSpace::DepthToSpaceMode> depthToSpaceModes = {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

/* *========================* Static Shapes Tests *========================* */

namespace static_shapes {

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {"jit_avx512"}, {"jit_avx512"}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, {"jit_avx512"}};

const auto cpuParams_nChw8c_avx2 = CPUSpecificParams {{nChw8c}, {nChw8c}, {"jit_avx2"}, {"jit_avx2"}};
const auto cpuParams_nCdhw8c_avx2 = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, {"jit_avx2"}};

const auto cpuParams_nChw8c_sse42 = CPUSpecificParams {{nChw8c}, {nChw8c}, {"jit_sse42"}, {"jit_sse42"}};
const auto cpuParams_nCdhw8c_sse42 = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, {"jit_sse42"}};

const auto cpuParams_nhwc_avx2 = CPUSpecificParams {{nhwc}, {nhwc}, {"jit_avx2"}, {"jit_avx2"}};
const auto cpuParams_ndhwc_avx2 = CPUSpecificParams {{ndhwc}, {ndhwc}, {"jit_avx2"}, {"jit_avx2"}};

const auto cpuParams_nhwc_sse42 = CPUSpecificParams {{nhwc}, {nhwc}, {"jit_sse42"}, {"jit_sse42"}};
const auto cpuParams_ndhwc_sse42 = CPUSpecificParams {{ndhwc}, {ndhwc}, {"jit_sse42"}, {"jit_sse42"}};

const auto cpuParams_nhwc_ref = CPUSpecificParams {{nhwc}, {nhwc}, {"ref_any"}, {"ref_any"}};
const auto cpuParams_ndhwc_ref = CPUSpecificParams {{ndhwc}, {ndhwc}, {"ref_any"}, {"ref_any"}};

const std::vector<DepthToSpaceInputShapes> inputShapesBS2_4D = {
        {{}, {{1, 64,  1, 1}}},
        {{}, {{1, 64,  1, 3}}},
        {{}, {{1, 128, 3, 3}}},
        {{}, {{2, 128, 1, 1}}},
        {{}, {{1, 192, 2, 2}}},
        {{}, {{2, 256, 2, 3}}},
        {{}, {{1, 512, 2, 1}}},
};

const std::vector<DepthToSpaceInputShapes> inputShapesBS3_4D = {
        {{}, {{1, 27, 1, 1}}},
        {{}, {{1, 27, 2, 3}}},
        {{}, {{1, 18, 2, 3}}},
        {{}, {{3, 18, 1, 1}}},
        {{}, {{2, 18, 3, 1}}},
};

const std::vector<CPUSpecificParams> CPUParamsBS2_4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c_avx2,
        cpuParams_nChw8c_sse42,
        cpuParams_nhwc_avx2,
        cpuParams_nhwc_sse42,
        cpuParams_nhwc_ref,
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS2_4D, DepthToSpaceLayerCPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapesBS2_4D),
                                 testing::ValuesIn(inputPrecisions),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2),
                                 testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS2_4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParamsBS3_4D = {
        cpuParams_nhwc_avx2,
        cpuParams_nhwc_sse42,
        cpuParams_nhwc_ref,
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS3_4D, DepthToSpaceLayerCPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapesBS3_4D),
                                 testing::ValuesIn(inputPrecisions),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 3),
                                 testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS3_4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<DepthToSpaceInputShapes> inputShapesBS2_5D = {
        {{}, {{1, 128, 1, 1, 1}}},
        {{}, {{1, 128, 2, 1, 2}}},
        {{}, {{1, 256, 2, 1, 3}}},
        {{}, {{2, 256, 3, 1, 1}}},
        {{}, {{1, 384, 1, 2, 2}}},
        {{}, {{2, 512, 1, 2, 1}}},
};

const std::vector<DepthToSpaceInputShapes> inputShapesBS3_5D = {
        {{}, {{1, 54, 1, 1, 1}}},
        {{}, {{1, 54, 2, 1, 2}}},
        {{}, {{3, 54, 1, 1, 1}}},
        {{}, {{2, 54, 3, 1, 2}}},
        {{}, {{1, 54, 3, 2, 2}}},
};

const std::vector<CPUSpecificParams> CPUParamsBS2_5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c_avx2,
        cpuParams_nCdhw8c_sse42,
        cpuParams_ndhwc_avx2,
        cpuParams_ndhwc_sse42,
        cpuParams_ndhwc_ref,
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS2_5D, DepthToSpaceLayerCPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapesBS2_5D),
                                 testing::ValuesIn(inputPrecisions),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2),
                                 testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS2_5D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParamsBS3_5D = {
        cpuParams_ndhwc_avx2,
        cpuParams_ndhwc_sse42,
        cpuParams_ndhwc_ref,
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceStaticBS3_5D, DepthToSpaceLayerCPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapesBS3_5D),
                                 testing::ValuesIn(inputPrecisions),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 3),
                                 testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS3_5D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

} // namespace static_shapes
/* *========================* *==================* *========================* */


/* *========================* Dynamic Shapes Tests *========================* */
namespace dynamic_shapes {

const auto cpuParams_avx512 = CPUSpecificParams {{}, {}, {"jit_avx512"}, {"jit_avx512"}};
const auto cpuParams_avx2 = CPUSpecificParams {{}, {}, {"jit_avx2"}, {"jit_avx2"}};
const auto cpuParams_sse42 = CPUSpecificParams {{}, {}, {"jit_sse42"}, {"jit_sse42"}};
const auto cpuParams_ref = CPUSpecificParams {{}, {}, {"ref_any"}, {"ref_any"}};

const std::vector<DepthToSpaceInputShapes> inputShapes4D = {
        {{{-1, -1, -1 , -1}},                                // dynamic
         {{2, 36, 1, 1}, {1, 36, 3, 1}, {1, 72, 1, 4}}},     // target

        {{{-1, 216, -1 , -1}},                               // dynamic
         {{1, 216, 1, 1}, {1, 216, 2, 2}, {3, 216, 4, 1}}},  // target

        {{{{1, 5}, {36, 72}, {1, 16}, {1, 16}}},             // dynamic
         {{3, 36, 4, 4}, {1, 36, 16, 12}, {3, 72, 8, 8}}},   // target
};

const std::vector<DepthToSpaceInputShapes> inputShapes5D = {
        {{{-1, -1, -1 , -1, -1}},                                     // dynamic
         {{2, 216, 1, 1, 1}, {1, 216, 3, 1, 2}, {1, 432, 2, 3, 1}}},  // target

        {{{-1, 216, -1 , -1, -1}},                                    // dynamic
         {{1, 216, 1, 1, 1}, {1, 216, 2, 1, 4}, {3, 216, 4, 1, 2}}},  // target

        {{{{1, 3}, {216, 432}, {1, 4}, {1, 4}, {1, 4}}},              // dynamic
         {{3, 216, 2, 2, 2}, {1, 432, 1, 1, 1}}},                     // target
};

const std::vector<CPUSpecificParams> CPUParams = {
        cpuParams_avx512,
        cpuParams_avx2,
        cpuParams_sse42,
        cpuParams_ref,
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamic4D, DepthToSpaceLayerCPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes4D),
                                 testing::ValuesIn(inputPrecisions),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2, 3),
                                 testing::ValuesIn(filterCPUInfoForDevice(CPUParams))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamic5D, DepthToSpaceLayerCPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes5D),
                                 testing::ValuesIn(inputPrecisions),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2, 3),
                                 testing::ValuesIn(filterCPUInfoForDevice(CPUParams))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

} // namespace dynamic_shapes
/* *========================* *==================* *========================* */

} // namespace
} // namespace CPULayerTestsDefinitions
