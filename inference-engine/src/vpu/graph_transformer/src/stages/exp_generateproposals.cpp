// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vpu/frontend/frontend.hpp>

#include <memory>

namespace vpu {

namespace {

VPU_PACKED(ExpGenerateProposalsParams {
    float   min_size;
    float   nms_threshold;
    int32_t pre_nms_topn;
    int32_t post_nms_topn;
};)

class ExpGenerateProposalsStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpGenerateProposalsStage>(*this);
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
        assertInputsOutputsTypes(this,
             {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}, {DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<ExpGenerateProposalsParams>("params");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        for (auto& inputEdge : inputEdges()) {
            inputEdge->input()->serializeBuffer(serializer);
        }

        for (auto& outputEdge : outputEdges()) {
            outputEdge->output()->serializeBuffer(serializer);
        }
    }
};

}  // namespace

void FrontEnd::parseExpGenerateProposals(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 4, "Layer %s must have 4 input tensors.", layer->name);
    VPU_THROW_UNLESS(outputs.size() == 2, "Layer %s must have 2 output tensors.", layer->name);

    ExpGenerateProposalsParams params;

    params.min_size      = layer->GetParamAsFloat("min_size", 0.0f);
    params.nms_threshold = layer->GetParamAsFloat("nms_threshold", 0.7f);
    params.pre_nms_topn  = layer->GetParamAsInt("pre_nms_count", 1000);
    params.post_nms_topn = layer->GetParamAsInt("post_nms_count", 1000);

    auto imInfo       = inputs[0];
    auto inputAnchors = inputs[1];
    auto inputDeltas  = inputs[2];
    auto inputScores  = inputs[3];
    auto outputRois   = outputs[0];
    auto outputScores = outputs[1];

    VPU_THROW_UNLESS((inputAnchors->desc().dims().size() == 2) &&
                     (inputAnchors->desc().dim(Dim::C) == 4),
                     "Wrong shape for input 1 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     layer->name, inputAnchors->desc().dims().size(), inputAnchors->desc().dim(Dim::C));
    VPU_THROW_UNLESS((imInfo->desc().dims().size() == 1) &&
                     (imInfo->desc().dim(Dim::C) == 3),
                     "Wrong shape for input 0 of layer %s, expected (3), got: dims size = %lu, dim C = %d",
                     layer->name, imInfo->desc().dims().size(), imInfo->desc().dim(Dim::C));

    VPU_THROW_UNLESS(inputDeltas->desc().dims().size() == 3,
                     "Wrong shape for input 2 of layer %s, expected dim size = 3, got: %lu",
                     layer->name, inputDeltas->desc().dims().size());
    VPU_THROW_UNLESS(inputScores->desc().dims().size() == 3,
                     "Wrong shape for input 3 of layer %s, expected dim size = 3, got: %lu",
                     layer->name, inputScores->desc().dims().size());

    VPU_THROW_UNLESS((inputDeltas->desc().dim(Dim::H) == inputScores->desc().dim(Dim::H)) &&
                     (inputDeltas->desc().dim(Dim::W) == inputScores->desc().dim(Dim::W)),
                     "Inputs 2 and 3 of layer %s must have same H and W, got: input2 (H = %d, W = %d), input3 (H = %d, W = %d)",
                     layer->name, inputDeltas->desc().dim(Dim::H), inputDeltas->desc().dim(Dim::W),
                     inputScores->desc().dim(Dim::H), inputScores->desc().dim(Dim::W));

    VPU_THROW_UNLESS((outputRois->desc().dims().size() == 2) &&
                     (outputRois->desc().dim(Dim::C) == 4),
                     "Wrong shape for output 0 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     layer->name, outputRois->desc().dims().size(), outputRois->desc().dim(Dim::C));
    VPU_THROW_UNLESS(outputScores->desc().dims().size() == 1,
                     "Wrong shape for output 1 of layer %s, expected dim size = 1, got: %lu",
                     layer->name, outputScores->desc().dims().size());

    VPU_THROW_UNLESS(outputRois->desc().dim(Dim::N) == outputScores->desc().dim(Dim::C),
                     "Layer %s: output0 dim N and output1 dim C must be equal, got: output0 (N = %d), output1 (C = %d)",
                     layer->name, outputRois->desc().dim(Dim::N), outputScores->desc().dim(Dim::C));

    auto stage = model->addNewStage<ExpGenerateProposalsStage>(
        layer->name,
        StageType::ExpGenerateProposals,
        layer,
        inputs,
        outputs);

    stage->attrs().set("params", params);
}

}  // namespace vpu
