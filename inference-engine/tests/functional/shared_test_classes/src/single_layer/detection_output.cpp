// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/single_layer/detection_output.hpp"

namespace LayerTestsDefinitions {

using namespace ov::test;

std::string DetectionOutputLayerTest::getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
    DetectionOutputAttributes commonAttrs;
    ParamsWhichSizeDepends specificAttrs;
    ngraph::op::DetectionOutputAttrs attrs;
    size_t batch;
    bool replaceDynamicShapesToIntervals;
    std::string targetDevice;
    std::tie(commonAttrs, specificAttrs, batch, attrs.objectness_score, replaceDynamicShapesToIntervals, targetDevice) = obj.param;

    std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
             attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = commonAttrs;

    const size_t numInputs = 5;
    std::vector<ov::test::InputShape> inShapes(numInputs);
    std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
             inShapes[idxLocation], inShapes[idxConfidence], inShapes[idxPriors], inShapes[idxArmConfidence], inShapes[idxArmLocation]) = specificAttrs;

    if (inShapes[idxArmConfidence].first.rank().get_length() == 0ul) {
        inShapes.resize(3);
    }

    for (size_t i = 0; i < inShapes.size(); i++) {
        inShapes[i].first[0] = batch;
    }

    std::ostringstream result;
    result << "IS = { ";
    result << "LOC=" << inShapes[0] << "_";
    result << "CONF=" << inShapes[1] << "_";
    result << "PRIOR=" << inShapes[2];
    if (inShapes.size() > 3) {
        result << "_ARM_CONF=" << inShapes[3] << "_";
        result << "ARM_LOC=" << inShapes[4] << " }_";
    }

    result << "Classes=" << attrs.num_classes << "_";
    result << "backgrId=" << attrs.background_label_id << "_";
    result << "topK="  << attrs.top_k << "_";
    result << "varEnc=" << attrs.variance_encoded_in_target << "_";
    result << "keepTopK=" << CommonTestUtils::vec2str(attrs.keep_top_k) << "_";
    result << "codeType=" << attrs.code_type << "_";
    result << "shareLoc=" << attrs.share_location << "_";
    result << "nmsThr=" << attrs.nms_threshold << "_";
    result << "confThr=" << attrs.confidence_threshold << "_";
    result << "clipAfterNms=" << attrs.clip_after_nms << "_";
    result << "clipBeforeNms=" << attrs.clip_before_nms << "_";
    result << "decrId=" << attrs.decrease_label_id << "_";
    result << "norm=" << attrs.normalized << "_";
    result << "inH=" << attrs.input_height << "_";
    result << "inW=" << attrs.input_width << "_";
    result << "OS=" << attrs.objectness_score << "_";
    result << "RDS=" << (replaceDynamicShapesToIntervals ? "true" : "false") << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

ParamsWhichSizeDepends DetectionOutputLayerTest::fromStatic(
    const bool varianceEncodedInTarget,
    const bool shareLocation,
    const bool normalized,
    const size_t inputHeight,
    const size_t inputWidth,
    const ov::Shape& locationInput,
    const ov::Shape& confidenceInput,
    const ov::Shape& priorsInput,
    const ov::Shape& armConfidence,
    const ov::Shape& armLocation) {
    return {
        varianceEncodedInTarget,
        shareLocation,
        normalized,
        inputHeight,
        inputWidth,
        {locationInput, {locationInput}},
        {confidenceInput, {confidenceInput}},
        {priorsInput, {priorsInput}},
        {armConfidence, {armConfidence}},
        {armLocation, {armLocation}}
    };
}

void DetectionOutputLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (auto i = 0ul; i < funcInputs.size(); ++i) {
        const auto &funcInput = funcInputs[i];
        InferenceEngine::Blob::Ptr blob;
        int32_t resolution = 1;
        uint32_t range = 1;
        if (i == 2) {
            if (attrs.normalized) {
                resolution = 100;
            } else {
                range = 10;
            }
        } else if (i == 1 || i == 3) {
            resolution = 1000;
        } else {
            resolution = 10;
        }

        auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], range, 0, resolution);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void DetectionOutputLayerTest::compare(
    const std::vector<ov::runtime::Tensor>& expectedTensors,
    const std::vector<ov::runtime::Tensor>& actualTensors) {
    ASSERT_EQ(expectedTensors.size(), actualTensors.size());

    for (auto i = 0; i < expectedTensors.size(); ++i) {
        auto expected = expectedTensors[i];
        auto actual = actualTensors[i];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        size_t expSize = 0;
        const float* expBuf = expected.data<const float>();
        for (size_t i = 0; i < expected.get_size(); i+=7) {
            if (expBuf[i] == -1)
                break;
            expSize += 7;
        }

        size_t actSize = 0;
        const float* actBuf = actual.data<const float>();
        for (size_t i = 0; i < actual.get_size(); i+=7) {
            if (actBuf[i] == -1)
                break;
            actSize += 7;
        }

        ASSERT_EQ(expSize, actSize);
    }

    ov::test::SubgraphBaseTest::compare(expectedTensors, actualTensors);
}

void DetectionOutputLayerTest::SetUp() {
    DetectionOutputAttributes commonAttrs;
    ParamsWhichSizeDepends specificAttrs;
    size_t batch;
    bool replaceDynamicShapesToIntervals;
    std::tie(commonAttrs, specificAttrs, batch, attrs.objectness_score, replaceDynamicShapesToIntervals, targetDevice) = this->GetParam();

    std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
             attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = commonAttrs;

    inShapes.resize(numInputs);
    std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
             inShapes[idxLocation], inShapes[idxConfidence], inShapes[idxPriors], inShapes[idxArmConfidence], inShapes[idxArmLocation]) = specificAttrs;

    if (inShapes[idxArmConfidence].first.rank().get_length() == 0) {
        inShapes.resize(3);
    }

    if (replaceDynamicShapesToIntervals) {
        ov::test::utils::set_dimension_intervals(inShapes);
    }

    for (auto& value : inShapes) {
        auto shapes = value.second;
        for (auto& shape : shapes) {
            shape[0] = batch;
        }
    }

    init_input_shapes({ inShapes });

    auto params = ngraph::builder::makeDynamicParams(ngraph::element::f32, inputDynamicShapes);
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
    auto detOut = ngraph::builder::makeDetectionOutput(paramOuts, attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(detOut)};
    function = std::make_shared<ngraph::Function>(results, params, "DetectionOutput");
}
}  // namespace LayerTestsDefinitions
