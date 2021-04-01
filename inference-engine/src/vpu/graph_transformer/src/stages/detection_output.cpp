// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <memory>

#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

enum PriorBox_CodeType {
    CORNER = 1,
    CENTER_SIZE,
    CORNER_SIZE
};

VPU_PACKED(DetectionOutputParams {
    int32_t num_classes;
    int32_t share_location;
    int32_t background_label_id;
    float nms_threshold;
    int32_t top_k;
    int32_t code_type;
    int32_t keep_top_k;
    float confidence_threshold;
    int32_t variance_encoded_in_target;
    int32_t num_priors;
    int32_t clip_before_nms;
    int32_t clip_after_nms;
    int32_t decrease_label_id;
    int32_t image_width;
    int32_t image_height;
    int32_t normalized;
    int32_t num;
    float objectness_score;
    int32_t has_arm_inputs;
};)

void printTo(std::ostream& os, const DetectionOutputParams& params) {
    os << "[" << std::endl;
    os << "num_classes=" << params.num_classes << std::endl;
    os << "share_location=" << params.share_location << std::endl;
    os << "background_label_id=" << params.background_label_id << std::endl;
    os << "nms_threshold=" << params.nms_threshold << std::endl;
    os << "top_k=" << params.top_k << std::endl;
    os << "code_type=" << params.code_type << std::endl;
    os << "keep_top_k=" << params.keep_top_k << std::endl;
    os << "confidence_threshold=" << params.confidence_threshold << std::endl;
    os << "variance_encoded_in_target=" << params.variance_encoded_in_target << std::endl;
    os << "num_priors=" << params.num_priors << std::endl;
    os << "clip_before_nms=" << params.clip_before_nms << std::endl;
    os << "clip_after_nms=" << params.clip_after_nms << std::endl;
    os << "decrease_label_id=" << params.decrease_label_id << std::endl;
    os << "image_width=" << params.image_width << std::endl;
    os << "image_height=" << params.image_height << std::endl;
    os << "normalized=" << params.normalized << std::endl;
    os << "num=" << params.num << std::endl;
    os << "objectness_score=" << params.objectness_score << std::endl;
    os << "has_arm_inputs=" << params.has_arm_inputs << std::endl;
    os << "]";
}

void printTo(DotLabel& lbl, const DetectionOutputParams& params) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("num_classes", params.num_classes);
    subLbl.appendPair("share_location", params.share_location);
    subLbl.appendPair("background_label_id", params.background_label_id);
    subLbl.appendPair("nms_threshold", params.nms_threshold);
    subLbl.appendPair("top_k", params.top_k);
    subLbl.appendPair("code_type", params.code_type);
    subLbl.appendPair("keep_top_k", params.keep_top_k);
    subLbl.appendPair("confidence_threshold", params.confidence_threshold);
    subLbl.appendPair("variance_encoded_in_target", params.variance_encoded_in_target);
    subLbl.appendPair("num_priors", params.num_priors);
    subLbl.appendPair("clip_before_nms", params.clip_before_nms);
    subLbl.appendPair("clip_after_nms", params.clip_after_nms);
    subLbl.appendPair("decrease_label_id", params.decrease_label_id);
    subLbl.appendPair("image_width", params.image_width);
    subLbl.appendPair("image_height", params.image_height);
    subLbl.appendPair("normalized", params.normalized);
    subLbl.appendPair("num", params.num);
    subLbl.appendPair("objectness_score", params.objectness_score);
    subLbl.appendPair("has_arm_inputs", params.has_arm_inputs);
}

class DetectionOutputStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<DetectionOutputStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 3 || numInputs() == 5);
        IE_ASSERT(numOutputs() == 1);
        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<DetectionOutputParams>("params");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto loc = inputEdge(0)->input();
        auto conf = inputEdge(1)->input();
        auto priors = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        loc->serializeBuffer(serializer);
        conf->serializeBuffer(serializer);
        priors->serializeBuffer(serializer);
        if (numInputs() == 5) {
            inputEdge(3)->input()->serializeBuffer(serializer);
            inputEdge(4)->input()->serializeBuffer(serializer);
        }
        output->serializeBuffer(serializer);

        tempBuffer(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseDetectionOutput(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 3 || inputs.size() == 5);
    IE_ASSERT(outputs.size() == 1);

    auto loc = inputs[0];
    auto conf = inputs[1];
    auto priors = inputs[2];

    DetectionOutputParams detParams;
    detParams.num_classes = layer->GetParamAsInt("num_classes", 0);
    detParams.background_label_id = layer->GetParamAsInt("background_label_id", 0);
    detParams.top_k = layer->GetParamAsInt("top_k", -1);
    detParams.variance_encoded_in_target = layer->GetParamAsInt("variance_encoded_in_target", 0);
    detParams.keep_top_k = layer->GetParamAsInt("keep_top_k", -1);
    detParams.nms_threshold = layer->GetParamAsFloat("nms_threshold", 0);
    detParams.confidence_threshold = layer->GetParamAsFloat("confidence_threshold", -1.0f);
    detParams.share_location = layer->GetParamAsInt("share_location", 1);
    detParams.clip_before_nms = layer->GetParamAsInt("clip_before_nms", 0) || layer->GetParamAsInt("clip", 0);
    detParams.clip_after_nms = layer->GetParamAsInt("clip_after_nms", 0);
    detParams.decrease_label_id = layer->GetParamAsInt("decrease_label_id", 0);
    detParams.normalized = layer->GetParamAsInt("normalized", 1);
    detParams.image_height = layer->GetParamAsInt("input_height", 1);
    detParams.image_width = layer->GetParamAsInt("input_width", 1);
    detParams.objectness_score = layer->GetParamAsFloat("objectness_score", -1.0f);
    detParams.has_arm_inputs = inputs.size() == 5 ? 1 : 0;

    int prior_size = detParams.normalized ? 4 : 5;
    int num_loc_classes = detParams.share_location ? 1 : detParams.num_classes;

    detParams.num_priors = static_cast<int>(priors->desc().dim(Dim::W) / prior_size);
    detParams.num = static_cast<int>(conf->desc().dim(Dim::N));

    auto code_type_str = layer->GetParamAsString("code_type", "caffe.PriorBoxParameter.CENTER_SIZE");
    if (code_type_str.find("CORNER_SIZE") != std::string::npos) {
        detParams.code_type = CORNER_SIZE;
    } else if (code_type_str.find("CENTER_SIZE") != std::string::npos) {
        detParams.code_type = CENTER_SIZE;
    } else if (code_type_str.find("CORNER") != std::string::npos) {
        detParams.code_type = CORNER;
    } else {
        VPU_THROW_EXCEPTION << "Unknown code_type " << code_type_str << " for DetectionOutput layer " << layer->name;
    }

    if (detParams.keep_top_k < 0)
        detParams.keep_top_k = outputs[0]->desc().dim(Dim::H);

    if (detParams.num_priors * num_loc_classes * 4 != loc->desc().dim(Dim::C))
        VPU_THROW_EXCEPTION << "Detection Output: Number of priors must match number of location predictions.";

    if (detParams.num_priors * detParams.num_classes != conf->desc().dim(Dim::C))
        VPU_THROW_EXCEPTION << "Detection Output: Number of priors must match number of confidence predictions.";

    if (detParams.decrease_label_id && detParams.background_label_id != 0)
        VPU_THROW_EXCEPTION << "Detection Output: Cannot use decrease_label_id and background_label_id parameter simultaneously.";

    if (outputs[0]->desc().dim(Dim::H) < detParams.keep_top_k)
        VPU_THROW_EXCEPTION << "Detection Output: Output size more than output tensor.";

    if (outputs[0]->desc().dim(Dim::W) != 7)
        VPU_THROW_EXCEPTION << "Detection Output: Support only 7 vals per detection.";

    auto stage = model->addNewStage<DetectionOutputStage>(layer->name, StageType::DetectionOutput, layer, inputs, outputs);

    stage->attrs().set("params", detParams);

    int _num = detParams.num;
    int _num_classes = detParams.num_classes;
    int _num_priors = detParams.num_priors;
    int ALIGN_VALUE = 64;

    int size_decoded_bboxes_buf    = sizeof(int16_t)*_num*_num_classes*_num_priors*4 + ALIGN_VALUE;
    int size_buffer_buf            = sizeof(int32_t)*_num*_num_classes*_num_priors + ALIGN_VALUE;
    int size_indices_buf           = sizeof(int32_t)*_num*_num_classes*_num_priors + ALIGN_VALUE;
    int size_detections_count_buf  = sizeof(int32_t)*_num*_num_classes + ALIGN_VALUE;
    int size_reordered_conf_buf    = sizeof(int16_t)     *_num_classes*_num_priors + ALIGN_VALUE;
    int size_bbox_sizes_buf        = sizeof(int16_t)*_num*_num_classes*_num_priors + ALIGN_VALUE;
    int size_num_priors_actual_buf = sizeof(int32_t)*_num + ALIGN_VALUE;
    int size_temp_data_buf         = sizeof(int16_t)*env.resources.numSHAVEs*(_num_priors+8)*5 + ALIGN_VALUE;

    int buffer_size =
        size_decoded_bboxes_buf +
        size_buffer_buf +
        size_indices_buf +
        size_detections_count_buf +
        size_reordered_conf_buf +
        size_bbox_sizes_buf +
        size_num_priors_actual_buf +
        size_temp_data_buf;

    model->addTempBuffer(stage, buffer_size);
}

}  // namespace vpu
