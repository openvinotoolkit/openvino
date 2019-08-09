// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>

#include <vpu/utils/extra.hpp>

namespace vpu {

namespace {

class ShrinkStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ShrinkStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        VPU_THROW_EXCEPTION << "Must never be called";
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

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto dimsOrder = input->desc().dimsOrder();

        //
        // Get smallest Dim over which Shrink is done.
        //

        auto minShrinkDimInd = dimsOrder.numDims();

        for (const auto& p : input->desc().dims()) {
            if (output->desc().dim(p.first) != p.second) {
                minShrinkDimInd = std::min(minShrinkDimInd, dimsOrder.dimInd(p.first));
            }
        }

        //
        // Initial StridesRequirement for inputs and output.
        //

        auto inputReqs = input->requiredStrides();

        auto outputReqs = inputReqs;

        //
        // Merge output consumers StridesRequirement.
        //

        for (const auto& consumerEdge : output->consumerEdges()) {
            const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

            if (consumerInfo.hasInput(consumerEdge)) {
                const auto& consumerReqs = consumerInfo.getInput(consumerEdge);

                for (int i = 0; i < dimsOrder.numDims(); ++i) {
                    if (inputReqs.get(i) == DimStride::Any) {
                        if (consumerReqs.get(i) != DimStride::Any) {
                            inputReqs.add(i, consumerReqs.get(i));
                            outputReqs.add(i, consumerReqs.get(i));
                        }
                    }
                }
            }
        }

        //
        // Remove extra output StridesRequirement.
        //

        for (int i = minShrinkDimInd + 1; i < dimsOrder.numDims(); ++i) {
            outputReqs.remove(i);
        }

        //
        // Return merged StridesRequirements.
        //

        _stridesInfo.setInput(_inputEdges[0], inputReqs);
        _stridesInfo.setOutput(_outputEdges[0], outputReqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

Stage StageBuilder::addShrinkStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const DimValues& offset) {
    auto stage = model->addNewStage<ShrinkStage>(
        name,
        StageType::Shrink,
        layer,
        {input},
        {output});

    stage->attrs().set<DimValues>("offset", offset);

    return stage;
}

}  // namespace vpu
