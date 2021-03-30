// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/tile.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using TileLayerTestParamsSet = LayerTestsDefinitions::TileLayerTestParamsSet;
using TileSpecificParams = LayerTestsDefinitions::TileSpecificParams;

typedef std::tuple<
        TileLayerTestParamsSet,
        CPUSpecificParams> TileLayerCPUTestParamsSet;

class TileLayerCPUTest : public testing::WithParamInterface<TileLayerCPUTestParamsSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TileLayerCPUTestParamsSet> obj) {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::TileLayerTest::getTestCaseName(testing::TestParamInfo<TileLayerTestParamsSet>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        TileSpecificParams tileParams;
        std::vector<size_t> inputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(tileParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto repeats = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, std::vector<size_t>{tileParams.size()}, tileParams);
        auto tile = std::make_shared<ngraph::opset1::Tile>(params[0], repeats);
        tile->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(tile)};
        function = std::make_shared<ngraph::Function>(results, params, "tile");
    }

//    std::vector<cpu_memory_format_t> inFmts, outFmts;
//    std::vector<std::string> priority;
//    std::string selectedType;
};

TEST_P(TileLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Tile");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, "ref"};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "ref"};

const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* PARAMS */
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP32
};

const std::vector<std::vector<size_t>> inputShapes4D = {
        {2, 16, 3, 4},
        {1, 16, 1, 1},
};

const std::vector<std::vector<int64_t>> repeats4D = {
        {2, 3},
        {1, 2, 3},
        {1, 1, 1, 1},
        {1, 1, 2, 3},
        {1, 2, 1, 3},
        {2, 1, 1, 1},
        {2, 3, 1, 1},
};

const std::vector<std::vector<int64_t>> repeats5D = {
        {1, 2, 3},
        {1, 1, 2, 3},
        {1, 1, 1, 2, 3},
        {1, 2, 1, 1, 3},
        {2, 1, 1, 1, 1},
        {2, 3, 1, 1, 1},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nchw,
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc,
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_ncdhw,
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};
/* ============= */

/* INSTANCES */
INSTANTIATE_TEST_CASE_P(smoke_Tile_4D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(repeats4D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes4D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams4D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Tile_5D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(repeats5D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 16, 2, 3, 4})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams5D)),
                        TileLayerCPUTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPULayerTestsDefinitions
