// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>

namespace vpu {

namespace {

class CTCDecoderStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CTCDecoderStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>&,
            ScalePropagationStep) override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        out[input0] = 1.0f;
        out[input1] = 1.0f;
        out[output] = 1.0f;

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<DimsOrder> out;

        auto cInd = input->desc().dimsOrder().dimInd(Dim::C);
        out[output] = output->desc().dimsOrder().createMovedDim(Dim::C, cInd);

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<StridesRequirement> out;

        out[input] = StridesRequirement::compact();
        out[output] = StridesRequirement::compact();

        return out;
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<BatchSupport> out;

        out[input] = BatchSupport::Split;
        out[output] = BatchSupport::Split;

        return out;
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeOldBuffer(handle_from_this(), serializer);
        input1->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseCTCDecoder(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto ctc_merge_repeated_ = layer->GetParamAsInt("ctc_merge_repeated", 1);
    if (ctc_merge_repeated_ != 1) {
        VPU_THROW_EXCEPTION
            << layer->name <<  " [" << layer->type
            << "] has incorrect ctc_merge_repeated param value."
            << " Kernel support case when ctc_merge_repeated_ == 1 only";
    }

    model->addNewStage<CTCDecoderStage>(
        layer->name,
        StageType::CTCDecoder,
        layer,
        inputs,
        outputs);
}

}  // namespace vpu
