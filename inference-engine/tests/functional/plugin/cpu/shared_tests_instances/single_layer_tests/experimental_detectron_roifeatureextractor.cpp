// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {

using Attrs = ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = ov::op::v6::ExperimentalDetectronROIFeatureExtractor;

using ExperimentalDetectronROIFeatureExtractorLayerTestCPUParams = std::tuple<
        std::vector<InputShape>,                // Input shapes
        int64_t,                                // Output size
        int64_t,                                // Sampling ratio
        std::vector<int64_t>,                   // Pyramid scales
        bool,                                   // Aligned
        Precision>;                             // Network precision

class ExperimentalDetectronROIFeatureExtractorLayerCPUTest : public testing::WithParamInterface<ExperimentalDetectronROIFeatureExtractorLayerTestCPUParams>,
                                                             public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronROIFeatureExtractorLayerTestCPUParams> &obj) {
        std::vector<InputShape> inputShapes;
        int64_t outputSize, samplingRatio;
        std::vector<int64_t> pyramidScales;
        bool aligned;
        Precision netPrecision;
        std::tie(inputShapes, outputSize, samplingRatio, pyramidScales, aligned, netPrecision) = obj.param;

        std::ostringstream result;
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto &shape : inputShapes) {
                result << CommonTestUtils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << "outputSize=" << outputSize << "_";
        result << "samplingRatio=" << samplingRatio << "_";
        result << "pyramidScales=" << CommonTestUtils::vec2str(pyramidScales) << "_";
        std::string alig = aligned ? "true" : "false";
        result << "aligned=" << alig << "_";
        result << "netPRC=" << netPrecision.name();
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        int64_t outputSize, samplingRatio;
        std::vector<int64_t> pyramidScales;
        bool aligned;
        Precision netPrecision;
        std::tie(inputShapes, outputSize, samplingRatio, pyramidScales, aligned, netPrecision) = this->GetParam();

        selectedType = std::string("ref_any_") + netPrecision.name();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        inType = outType = ngPrc;

        init_input_shapes(inputShapes);

        Attrs attrs;
        attrs.aligned = aligned;
        attrs.output_size = outputSize;
        attrs.sampling_ratio = samplingRatio;
        attrs.pyramid_scales = pyramidScales;

        auto params = ngraph::builder::makeDynamicParams(ngPrc, {inputDynamicShapes});
        auto paramsOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto experimentalDetectronROIFeatureExtractor = std::make_shared<ExperimentalROI>(paramsOuts, attrs);
        function = std::make_shared<ov::Function>(ov::OutputVector{experimentalDetectronROIFeatureExtractor->output(0),
                                                                          experimentalDetectronROIFeatureExtractor->output(1)},
                                                  "ExperimentalDetectronROIFeatureExtractor");
    }
};

TEST_P(ExperimentalDetectronROIFeatureExtractorLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    // TODO: Should be uncommented after updating the CheckPluginRelatedResults() method
    // CheckPluginRelatedResults(executableNetwork, "ExperimentalDetectronROIFeatureExtractor");
};

namespace {

const std::vector<std::vector<int64_t>> pyramidScales = {
        {8, 16, 32, 64},
        {4, 8, 16, 32},
        {2, 4, 8, 16}
};

const std::vector<std::vector<InputShape>> staticInputShape = {
        static_shapes_to_test_representation({{1000, 4}, {1, 8, 200, 336}, {1, 8, 100, 168}, {1, 8, 50, 84}, {1, 8, 25, 42}}),
        static_shapes_to_test_representation({{1000, 4}, {1, 16, 200, 336}, {1, 16, 100, 168}, {1, 16, 50, 84}, {1, 16, 25, 42}}),
        static_shapes_to_test_representation({{1200, 4}, {1, 8, 200, 42}, {1, 8, 100, 336}, {1, 8, 50, 168}, {1, 8, 25, 84}})
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_static, ExperimentalDetectronROIFeatureExtractorLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShape),
                                 ::testing::Values(14),
                                 ::testing::Values(2),
                                 ::testing::ValuesIn(pyramidScales),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(Precision::FP32)),
                         ExperimentalDetectronROIFeatureExtractorLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInputShape = {
        {
                {
                        {{-1, 4}, {{1000, 4}, {1500, 4}, {2000, 4}}},
                        {{1, 8, -1, -1}, {{1, 8, 200, 336}, {1, 8, 200, 42}, {1, 8, 200, 84}}},
                        {{1, 8, -1, -1}, {{1, 8, 100, 168}, {1, 8, 100, 336}, {1, 8, 25, 42}}},
                        {{1, 8, -1, -1}, {{1, 8, 50, 84}, {1, 8, 50, 168}, {1, 8, 100, 336}}},
                        {{1, 8, -1, -1}, {{1, 8, 25, 42}, {1, 8, 25, 84}, {1, 8, 50, 168}}}
                }
        },
        {
                {
                        {{-1, 4}, {{1000, 4}, {1100, 4}, {1200, 4}}},
                        {{1, {8, 16}, -1, -1}, {{1, 8, 200, 336}, {1, 12, 200, 336}, {1, 16, 200, 336}}},
                        {{1, {8, 16}, -1, -1}, {{1, 8, 100, 168}, {1, 12, 100, 168}, {1, 16, 100, 168}}},
                        {{1, {8, 16}, -1, -1}, {{1, 8, 50, 84}, {1, 12, 50, 84}, {1, 16, 50, 84}}},
                        {{1, {8, 16}, -1, -1}, {{1, 8, 25, 42}, {1, 12, 25, 42}, {1, 16, 25, 42}}}
                }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_dynamic, ExperimentalDetectronROIFeatureExtractorLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShape),
                                 ::testing::Values(14),
                                 ::testing::Values(2),
                                 ::testing::ValuesIn(pyramidScales),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(Precision::FP32)),
                         ExperimentalDetectronROIFeatureExtractorLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
