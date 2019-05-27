// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <memory>

namespace vpu {

namespace {

class PadStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PadStage>(*this);
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
        return DataMap<StridesRequirement>();
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<BatchSupport> out;

        // TODO: try merge with last dimension
        out[input] = BatchSupport::Split;
        out[output] = BatchSupport::Split;

        return out;
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        auto perm = input->desc().dimsOrder().toPermutation();
        IE_ASSERT(perm.size() <= 4);

        auto pad_value = attrs().get<float>("pad_value");
        auto pad_mode = attrs().get<PadMode>("pad_mode");
        const auto& pads_begin = attrs().get<DimValues>("pads_begin");
        const auto& pads_end = attrs().get<DimValues>("pads_end");

        int i = 0;
        for (; i < perm.size(); ++i) {
            serializer.append(static_cast<uint32_t>(pads_begin.get(perm[i], 0)));
            serializer.append(static_cast<uint32_t>(pads_end.get(perm[i], 0)));
        }
        for (; i < 4; ++i) {
            serializer.append(static_cast<uint32_t>(0));
            serializer.append(static_cast<uint32_t>(0));
        }

        serializer.append(static_cast<float>(pad_value));
        serializer.append(static_cast<uint32_t>(pad_mode));
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

void FrontEnd::parsePad(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::PadLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(layer->pads_begin.size() == 4);
    IE_ASSERT(layer->pads_end.size() == 4);

    DimValues pads_begin;
    pads_begin.set(Dim::W, layer->pads_begin[3]);
    pads_begin.set(Dim::H, layer->pads_begin[2]);
    pads_begin.set(Dim::C, layer->pads_begin[1]);
    pads_begin.set(Dim::N, layer->pads_begin[0]);

    DimValues pads_end;
    pads_end.set(Dim::W, layer->pads_end[3]);
    pads_end.set(Dim::H, layer->pads_end[2]);
    pads_end.set(Dim::C, layer->pads_end[1]);
    pads_end.set(Dim::N, layer->pads_end[0]);

    _stageBuilder->addPadStage(
        model,
        layer->name,
        layer,
        static_cast<PadMode>(layer->pad_mode),
        layer->pad_value,
        pads_begin,
        pads_end,
        inputs[0],
        outputs[0]);
}

Stage StageBuilder::addPadStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        PadMode padMode,
        float pad_value,
        const DimValues& pads_begin,
        const DimValues& pads_end,
        const Data& input,
        const Data& output) {
    auto stage = model->addNewStage<PadStage>(
        name,
        StageType::Pad,
        layer,
        {input},
        {output});

    stage->attrs().set<float>("pad_value", pad_value);
    stage->attrs().set<PadMode>("pad_mode", padMode);
    stage->attrs().set<DimValues>("pads_begin", pads_begin);
    stage->attrs().set<DimValues>("pads_end", pads_end);

    return stage;
}

}  // namespace vpu
