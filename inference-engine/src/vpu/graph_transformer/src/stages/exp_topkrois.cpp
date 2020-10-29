// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vpu/frontend/frontend.hpp>

#include <memory>

namespace vpu {

namespace {

class ExpTopKROIsStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpTopKROIsStage>(*this);
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
             {{DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<int32_t>("max_rois");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        input(0)->serializeBuffer(serializer);
        input(1)->serializeBuffer(serializer);
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseExpTopKROIs(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 2, "Layer %s must have 2 input tensors.", layer->name);
    VPU_THROW_UNLESS(outputs.size() == 1, "Layer %s must have 1 output tensor.", layer->name);

    int32_t max_rois = layer->GetParamAsInt("max_rois", 0);

    auto inputRois  = inputs[0];
    auto inputProbs = inputs[1];
    auto outputRois = outputs[0];

    VPU_THROW_UNLESS((inputRois->desc().dims().size() == 2) &&
                     (inputRois->desc().dim(Dim::C) == 4),
                     "Wrong shape for input 0 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     layer->name, inputRois->desc().dims().size(), inputRois->desc().dim(Dim::C));

    VPU_THROW_UNLESS(inputProbs->desc().dims().size() == 1,
                     "Wrong shape for input 1 of layer %s, expected dim size = 1, got: %lu",
                     layer->name, inputProbs->desc().dims().size());

    VPU_THROW_UNLESS(inputProbs->desc().dim(Dim::C) == inputRois->desc().dim(Dim::N),
                     "Layer %s: input0 dim N and input1 dim C must be equal, got: input0 (N = %d), input1 (C = %d)",
                     layer->name, inputProbs->desc().dim(Dim::N), inputProbs->desc().dim(Dim::C));

    VPU_THROW_UNLESS((outputRois->desc().dims().size() == 2) &&
                     (outputRois->desc().dim(Dim::C) == 4),
                     "Wrong shape for output 0 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     layer->name, outputRois->desc().dims().size(), outputRois->desc().dim(Dim::C));

    VPU_THROW_UNLESS(outputRois->desc().dim(Dim::N) == max_rois,
                     "Wrong shape for output 0 of layer %s, expected dim N = %d, got: dim N = %d",
                     layer->name, static_cast<int>(max_rois), outputRois->desc().dim(Dim::N));

    auto stage = model->addNewStage<ExpTopKROIsStage>(
        layer->name,
        StageType::ExpTopKROIs,
        layer,
        inputs,
        outputs);

    stage->attrs().set("max_rois", max_rois);
}

}  // namespace vpu
