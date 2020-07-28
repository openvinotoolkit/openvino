// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "common_test_utils/data_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "single_layer_tests/detection_output.hpp"

namespace LayerTestsDefinitions {

std::string DetectionOutputLayerTest::getTestCaseName(testing::TestParamInfo<DetectionOutputParams> obj) {
    DetectionOutputAttributes commonAttrs;
    ParamsWhichSizeDepends specificAttrs;
    ngraph::op::DetectionOutputAttrs attrs;
    size_t batch;
    std::string targetDevice;
    std::tie(commonAttrs, specificAttrs, batch, attrs.objectness_score, targetDevice) = obj.param;

    std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
             attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = commonAttrs;

    const size_t numInputs = 5;
    std::vector<InferenceEngine::SizeVector> inShapes(numInputs);
    std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
             inShapes[idxLocation], inShapes[idxConfidence], inShapes[idxPriors], inShapes[idxArmConfidence], inShapes[idxArmLocation]) = specificAttrs;

    if (inShapes[idxArmConfidence].empty()) {
        inShapes.resize(3);
    }

    for (size_t i = 0; i < inShapes.size(); i++) {
        inShapes[i][0] = batch;
    }

    std::ostringstream result;
    result << "IS = { ";
    result << "LOC=" << CommonTestUtils::vec2str(inShapes[0]) << "_";
    result << "CONF=" << CommonTestUtils::vec2str(inShapes[1]) << "_";
    result << "PRIOR=" << CommonTestUtils::vec2str(inShapes[2]);
    std::string armConf, armLoc;
    if (inShapes.size() > 3) {
        armConf = "_ARM_CONF=" + CommonTestUtils::vec2str(inShapes[3]) + "_";
        armLoc = "ARM_LOC=" + CommonTestUtils::vec2str(inShapes[4]);
    }
    result << armConf;
    result << armLoc << " }_";

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
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void DetectionOutputLayerTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    size_t it = 0;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;

        InferenceEngine::Blob::Ptr blob;
        if (it == 2) {
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            if (attrs.normalized) {
                CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, 1, 0, 100);
            } else {
                CommonTestUtils::fill_data_random<InferenceEngine::Precision::FP32>(blob, 10, 0, 1);
            }
        } else {
            int32_t resolution;
            if (it == 1 || it == 3) {
                resolution = 1000;
            } else {
                resolution = 10;
            }
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, 1, 0, resolution);
        }
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
        it++;
    }
    inferRequest.Infer();
}

void DetectionOutputLayerTest::Validate() {
    referenceDetectionOutput refDetOut(attrs, inShapes);
    std::vector<float> refOutput = refDetOut.run(inputs);
    const auto& actualOutputs = GetOutputs();
    const float *actualOutputData = actualOutputs[0]->cbuffer().as<const float *>();
    Compare<float>(refOutput.data(), actualOutputData, actualOutputs[0]->size(), 1e-2f);
}

void DetectionOutputLayerTest::SetUp() {
    DetectionOutputAttributes commonAttrs;
    ParamsWhichSizeDepends specificAttrs;
    size_t batch;
    std::tie(commonAttrs, specificAttrs, batch, attrs.objectness_score, targetDevice) = this->GetParam();

    std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
             attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = commonAttrs;

    inShapes.resize(numInputs);
    std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
             inShapes[idxLocation], inShapes[idxConfidence], inShapes[idxPriors], inShapes[idxArmConfidence], inShapes[idxArmLocation]) = specificAttrs;

    if (inShapes[idxArmConfidence].empty()) {
        inShapes.resize(3);
    }

    for (size_t i = 0; i < inShapes.size(); i++) {
        inShapes[i][0] = batch;
    }

    auto params = ngraph::builder::makeParams(ngraph::element::f32, inShapes);
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
    auto detOut = ngraph::builder::makeDetectionOutput(paramOuts, attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(detOut)};
    function = std::make_shared<ngraph::Function>(results, params, "DetectionOutput");
}

TEST_P(DetectionOutputLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions

