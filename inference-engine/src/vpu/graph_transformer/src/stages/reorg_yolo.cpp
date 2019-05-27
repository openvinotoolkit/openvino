// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class ReorgYoloStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReorgYoloStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        if (step == ScalePropagationStep::Propagate) {
            out[output] = inputScales.at(input);
        } else {
            // ReorgYolo can only propagate scaling.

            out[input] = 1.0f;
            out[output] = 1.0f;
        }

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inOrder = input->desc().dimsOrder();

        DataMap<DimsOrder> out;

        out[output] = inOrder;

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inOrder = input->desc().dimsOrder();

        DataMap<StridesRequirement> out;

        if (inOrder.dimInd(Dim::C) == 0) {
            out[input] = StridesRequirement::compact();
            out[output] = StridesRequirement::compact();
        }

        return out;
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<BatchSupport> out;

        out[input] = BatchSupport::Split;
        out[output] = BatchSupport::Split;

        return out;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto stride = attrs().get<int>("stride");

        serializer.append(static_cast<int32_t>(stride));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseReorgYolo(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ReorgYoloStage>(
        layer->name,
        StageType::ReorgYolo,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("stride", layer->GetParamAsInt("stride", 2));
}

}  // namespace vpu
