// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/experimental_detectron_roifeatureextractor.hpp"

namespace ov {
namespace test {
std::string ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronROIFeatureExtractorTestParams>& obj) {
    std::vector<InputShape> shapes;
    int64_t outputSize, sampling_ratio;
    std::vector<int64_t> pyramid_scales;
    bool aligned;
    ElementType model_type;
    std::string device_name;
    std::tie(shapes, outputSize, sampling_ratio, pyramid_scales, aligned, model_type, device_name) = obj.param;

    std::ostringstream result;
    if (shapes.front().first.size() != 0) {
        result << "IS=(";
        for (const auto &shape : shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result.seekp(-1, result.cur);
        result << ")_";
    }
    result << "TS=";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "outputSize=" << outputSize << "_";
    result << "sampling_ratio=" << sampling_ratio << "_";
    result << "pyramid_scales=" << ov::test::utils::vec2str(pyramid_scales) << "_";
    std::string alig = aligned ? "true" : "false";
    result << "aligned=" << alig << "_";
    result << "netPRC=" << model_type << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void ExperimentalDetectronROIFeatureExtractorLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    int64_t outputSize, sampling_ratio;
    std::vector<int64_t> pyramid_scales;
    bool aligned;
    ElementType model_type;
    std::string targetName;
    std::tie(shapes, outputSize, sampling_ratio, pyramid_scales, aligned, model_type, targetName) = this->GetParam();

    inType = outType = model_type;
    targetDevice = targetName;

    init_input_shapes(shapes);

    ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes attrs;
    attrs.aligned = aligned;
    attrs.output_size = outputSize;
    attrs.sampling_ratio = sampling_ratio;
    attrs.pyramid_scales = pyramid_scales;

    ov::ParameterVector params;
    ov::NodeVector inputs;
    for (auto&& shape : inputDynamicShapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(param);
        inputs.push_back(param);
    }
    auto experimentalDetectronROIFeatureExtractor = std::make_shared<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(inputs, attrs);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{experimentalDetectronROIFeatureExtractor->output(0), experimentalDetectronROIFeatureExtractor->output(1)},
        params,
        "ExperimentalDetectronROIFeatureExtractor");
}
} // namespace test
} // namespace ov
