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
            ScalePropagationStep step,
            StageDataInfo<float>& scaleInfo) override {
        if (step == ScalePropagationStep::Propagate) {
            scaleInfo.setOutput(outputEdge(0), inputScales[0]);
        } else {
            // Copy can only propagate scaling.
            scaleInfo.setInput(inputEdge(0), 1.0f);
            scaleInfo.setOutput(outputEdge(0), 1.0f);
        }
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement().remove(0));
        stridesInfo.setOutput(outputEdge(0), StridesRequirement().remove(0));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        const auto& type = input(0)->desc().type();
        assertInputsOutputsTypes(this, {{type}}, {{type}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

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
