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

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            _scaleInfo.setOutput(_outputEdges[0], inputScales[0]);
        } else {
            // Copy can only propagate scaling.
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _stridesInfo.setInput(_inputEdges[0], StridesRequirement().remove(0));
        _stridesInfo.setOutput(_outputEdges[0], StridesRequirement().remove(0));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
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
