// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class MVNStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<MVNStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto normalize = attrs().get<int>("normalize");
        auto across_channels = attrs().get<int>("across_channels");

        serializer.append(static_cast<int32_t>(normalize));
        serializer.append(static_cast<int32_t>(across_channels));
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

void FrontEnd::parseMVN(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::MVNLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    float def_eps = 1e-9f;
    float eps = layer->GetParamAsFloat("eps", def_eps);

    if (eps > 1e-7f) {
        VPU_THROW_EXCEPTION
            << "Layer " << layer->name << " [" << layer->type
            <<  "] in our kernel we use const value 1e-9f. Actual = " << eps;
    }

    auto stage = model->addNewStage<MVNStage>(
        layer->name,
        StageType::MVN,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("normalize", layer->normalize);
    stage->attrs().set<int>("across_channels", layer->across_channels);
}

}  // namespace vpu
