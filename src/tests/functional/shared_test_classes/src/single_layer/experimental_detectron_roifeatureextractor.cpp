// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/single_layer/experimental_detectron_roifeatureextractor.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronROIFeatureExtractorTestParams>& obj) {
    std::vector<InputShape> inputShapes;
    int64_t outputSize, samplingRatio;
    std::vector<int64_t> pyramidScales;
    bool aligned;
    ElementType netPrecision;
    std::string targetName;
    std::tie(inputShapes, outputSize, samplingRatio, pyramidScales, aligned, netPrecision, targetName) = obj.param;

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
    result << "netPRC=" << netPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronROIFeatureExtractorLayerTest::SetUp() {
    // TODO: Remove it after fixing issue 69529
    // w/a for myriad (cann't store 2 caches simultaneously)
    PluginCache::get().reset();

    std::vector<InputShape> inputShapes;
    int64_t outputSize, samplingRatio;
    std::vector<int64_t> pyramidScales;
    bool aligned;
    ElementType netPrecision;
    std::string targetName;
    std::tie(inputShapes, outputSize, samplingRatio, pyramidScales, aligned, netPrecision, targetName) = this->GetParam();

    inType[0] = outType[0] = netPrecision;
    targetDevice = targetName;

    init_input_shapes(inputShapes);

    Attrs attrs;
    attrs.aligned = aligned;
    attrs.output_size = outputSize;
    attrs.sampling_ratio = samplingRatio;
    attrs.pyramid_scales = pyramidScales;

    auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes});
    auto paramsOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto experimentalDetectronROIFeatureExtractor = std::make_shared<ExperimentalROI>(paramsOuts, attrs);
    function = std::make_shared<ov::Model>(ov::OutputVector{experimentalDetectronROIFeatureExtractor->output(0),
                                                               experimentalDetectronROIFeatureExtractor->output(1)},
                                              "ExperimentalDetectronROIFeatureExtractor");
}
} // namespace subgraph
} // namespace test
} // namespace ov
