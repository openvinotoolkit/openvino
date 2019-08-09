// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <memory>

namespace vpu {

namespace {

class NoneStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NoneStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        for (const auto& outEdge : _outputEdges) {
            _scaleInfo.setOutput(outEdge, 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
    }

    void getDataStridesRequirementsImpl() const override {
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

    void serializeDataImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addNoneStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    return model->addNewStage<NoneStage>(
        name,
        StageType::None,
        layer,
        {inputs},
        {outputs});
}

}  // namespace vpu
