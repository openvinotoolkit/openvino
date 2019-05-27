// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <unordered_set>
#include <memory>

namespace vpu {

void FrontEnd::parseCopy(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    _stageBuilder->addCopyStage(model, layer->name, layer, inputs[0], outputs[0]);
}

namespace {

class CopyStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<CopyStage>(*this);
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
            // Copy can only propagate scaling.
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

        DataMap<DimsOrder> out;

        out[output] = input->desc().dimsOrder();

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<StridesRequirement> out;

        out[input] = StridesRequirement().remove(0);
        out[output] = StridesRequirement().remove(0);

        return out;
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        if (input->desc().dimsOrder() == DimsOrder::NC) {
            if (!input->checkStrides(StridesRequirement().add(0, DimStride::Compact)) ||
                !output->checkStrides(StridesRequirement().add(0, DimStride::Compact))) {
                input->serializeOldBuffer(
                    handle_from_this(),
                    serializer,
                    DimsOrder::CHW,
                    {
                        {Dim::C, {Dim::N}},
                        {Dim::H, {Dim::C}},
                    });

                output->serializeOldBuffer(
                    handle_from_this(),
                    serializer,
                    DimsOrder::CHW,
                    {
                        {Dim::C, {Dim::N}},
                        {Dim::H, {Dim::C}},
                    });

                return;
            }
        }

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addCopyStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output) {
    return model->addNewStage<CopyStage>(
        name,
        StageType::Copy,
        layer,
        {input},
        {output});
}

}  // namespace vpu
