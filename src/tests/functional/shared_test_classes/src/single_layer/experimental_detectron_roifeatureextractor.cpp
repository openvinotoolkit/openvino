// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/single_layer/experimental_detectron_roifeatureextractor.hpp"
#include <random>

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

    inType = outType = netPrecision;
    targetDevice = targetName;
    mainPyramidScale = pyramidScales[0];

    init_input_shapes(inputShapes);

    Attrs attrs;
    attrs.aligned = aligned;
    attrs.output_size = outputSize;
    attrs.sampling_ratio = samplingRatio;
    attrs.pyramid_scales = pyramidScales;

    if (netPrecision == ElementType::bf16) {
        rel_threshold = 1e-2;
    }

    auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes});
    auto paramsOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto experimentalDetectronROIFeatureExtractor = std::make_shared<ExperimentalROI>(paramsOuts, attrs);
    function = std::make_shared<ov::Model>(ov::OutputVector{experimentalDetectronROIFeatureExtractor->output(0),
                                                               experimentalDetectronROIFeatureExtractor->output(1)},
                                              "ExperimentalDetectronROIFeatureExtractor");
}

void ExperimentalDetectronROIFeatureExtractorLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> tempRois(targetInputStaticShapes[0][0]);
    const size_t height = targetInputStaticShapes[1][2] * mainPyramidScale;
    const size_t width = targetInputStaticShapes[1][3] * mainPyramidScale;

    std::random_device rd;
    std::default_random_engine rg{rd()};
    std::uniform_int_distribution<uint32_t> coords(1, 30);
    std::vector<std::uniform_int_distribution<uint32_t>> sizes{
        std::uniform_int_distribution<uint32_t>{20, 110},  // pyramid level 1
        std::uniform_int_distribution<uint32_t>{120, 220}, // pyramid level 2
        std::uniform_int_distribution<uint32_t>{230, 440}, // pyramid level 3
        std::uniform_int_distribution<uint32_t>{450, 600}  // pyramid level 4
    };

    size_t sizesIndex = 0;
    for (auto& roi : tempRois) {
        auto x1 = coords(rg);
        auto y1 = coords(rg);
        auto x2 = x1 + sizes[sizesIndex](rg);
        auto y2 = y1 + sizes[sizesIndex](rg);
        if (x2 >= width)
            x2 = width - 1;
        if (y2 >= height)
            y2 = height - 1;

        std::get<0>(roi) = x1;
        std::get<1>(roi) = y1;
        std::get<2>(roi) = x2;
        std::get<3>(roi) = y2;

        if (sizesIndex == 3)
            sizesIndex = 0;
        else
            ++sizesIndex;
    }

    std::shuffle(tempRois.begin(), tempRois.end(), rg);

    const auto& prec = funcInputs[0].get_element_type();
    ov::Tensor roiTensor{ prec, targetInputStaticShapes[0] };
    if (prec == ElementType::f32) {
        auto roiTensorData = static_cast<float*>(roiTensor.data());
        for (size_t i = 0, roiIndex = 0; i < roiTensor.get_size(); i += 4, ++roiIndex) {
            roiTensorData[i] = std::get<0>(tempRois[roiIndex]);
            roiTensorData[i + 1] = std::get<1>(tempRois[roiIndex]);
            roiTensorData[i + 2] = std::get<2>(tempRois[roiIndex]);
            roiTensorData[i + 3] = std::get<3>(tempRois[roiIndex]);
        }
    } else if (prec == ElementType::bf16) {
        auto roiTensorData = static_cast<std::int16_t*>(roiTensor.data());
        for (size_t i = 0, roiIndex = 0; i < roiTensor.get_size(); i += 4, ++roiIndex) {
            roiTensorData[i] = static_cast<std::int16_t>(ngraph::bfloat16(std::get<0>(tempRois[roiIndex])).to_bits());
            roiTensorData[i + 1] = static_cast<std::int16_t>(ngraph::bfloat16(std::get<1>(tempRois[roiIndex])).to_bits());
            roiTensorData[i + 2] = static_cast<std::int16_t>(ngraph::bfloat16(std::get<2>(tempRois[roiIndex])).to_bits());
            roiTensorData[i + 3] = static_cast<std::int16_t>(ngraph::bfloat16(std::get<3>(tempRois[roiIndex])).to_bits());
        }
    } else {
        IE_THROW() << "ExperimentalDetectronROIFeatureExtractor: unsupported precision: " << prec;
    }

    inputs.insert({ funcInputs[0].get_node_shared_ptr(), roiTensor });

    for (size_t i = 1; i < funcInputs.size(); ++i) {
        const auto& dataPrecision = funcInputs[i].get_element_type();
        const auto& dataShape = targetInputStaticShapes[i];
        ov::Tensor dataTensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape);

        inputs.insert({ funcInputs[i].get_node_shared_ptr(), dataTensor });
    }
}

} // namespace subgraph
} // namespace test
} // namespace ov
