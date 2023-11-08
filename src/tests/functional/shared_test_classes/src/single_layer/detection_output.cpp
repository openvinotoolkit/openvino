// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/detection_output.hpp"

namespace LayerTestsDefinitions {

std::ostream& operator <<(std::ostream& result, const ngraph::op::DetectionOutputAttrs& attrs) {
    result << "Classes=" << attrs.num_classes << "_";
    result << "backgrId=" << attrs.background_label_id << "_";
    result << "topK="  << attrs.top_k << "_";
    result << "varEnc=" << attrs.variance_encoded_in_target << "_";
    result << "keepTopK=" << ov::test::utils::vec2str(attrs.keep_top_k) << "_";
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
    return result;
}

std::string DetectionOutputLayerTest::getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
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
    result << "LOC=" << ov::test::utils::vec2str(inShapes[0]) << "_";
    result << "CONF=" << ov::test::utils::vec2str(inShapes[1]) << "_";
    result << "PRIOR=" << ov::test::utils::vec2str(inShapes[2]);
    std::string armConf, armLoc;
    if (inShapes.size() > 3) {
        armConf = "_ARM_CONF=" + ov::test::utils::vec2str(inShapes[3]) + "_";
        armLoc = "ARM_LOC=" + ov::test::utils::vec2str(inShapes[4]);
    }
    result << armConf;
    result << armLoc << " }_";

    result << attrs;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void DetectionOutputLayerTest::GenerateInputs() {
    size_t it = 0;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        InferenceEngine::Blob::Ptr blob;
        int32_t resolution = 1;
        uint32_t range = 1;
        if (it == 2) {
            if (attrs.normalized) {
                resolution = 100;
            } else {
                range = 10;
            }
        } else if (it == 1 || it == 3) {
            resolution = 1000;
        } else {
            resolution = 10;
        }
        blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();
        ov::test::utils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, range, 0, resolution);
        inputs.push_back(blob);
        it++;
    }
}

void DetectionOutputLayerTest::Compare(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
        const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto &expected = expectedOutputs[outputIndex].second;
        const auto &actual = actualOutputs[outputIndex];

        ASSERT_EQ(expected.size(), actual->byteSize());

        size_t expSize = 0;
        size_t actSize = 0;

        const auto &expectedBuffer = expected.data();
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

        const float *expBuf = reinterpret_cast<const float *>(expectedBuffer);
        const float *actBuf = reinterpret_cast<const float *>(actualBuffer);
        for (size_t i = 0; i < actual->size(); i+=7) {
            if (expBuf[i] == -1)
                break;
            expSize += 7;
        }
        for (size_t i = 0; i < actual->size(); i+=7) {
            if (actBuf[i] == -1)
                break;
            actSize += 7;
        }
        ASSERT_EQ(expSize, actSize);
        LayerTestsCommon::Compare<float>(expBuf, actBuf, expSize, 1e-2f);
    }
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

    ov::ParameterVector params;
    for (auto&& shape : inShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape)));
    }
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
    auto detOut = ngraph::builder::makeDetectionOutput(paramOuts, attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(detOut)};
    function = std::make_shared<ngraph::Function>(results, params, "DetectionOutput");
}
}  // namespace LayerTestsDefinitions
