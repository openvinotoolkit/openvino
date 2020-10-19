// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <utility>

namespace vpu {

namespace {

class SplitStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<SplitStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        for (const auto& outEdge : outputEdges()) {
            orderInfo.setOutput(outEdge, input->desc().dimsOrder());
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto input = inputEdge(0)->input();

        auto dimsOrder = input->desc().dimsOrder();

        //
        // Get smallest Dim over which Split is done.
        //

        auto minSplitDimInd = dimsOrder.numDims();

        for (const auto& outEdge : outputEdges()) {
            auto output = outEdge->output();

            for (const auto& p : input->desc().dims()) {
                if (output->desc().dim(p.first) != p.second) {
                    minSplitDimInd = std::min(minSplitDimInd, dimsOrder.dimInd(p.first));
                }
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

        for (const auto& outEdge : outputEdges()) {
            auto curOutput = outEdge->output();

            for (const auto& consumerEdge : curOutput->consumerEdges()) {
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
        }

        //
        // Remove extra output StridesRequirement.
        //

        for (int i = minSplitDimInd + 1; i < dimsOrder.numDims(); ++i) {
            outputReqs.remove(i);
        }

        //
        // Return merged StridesRequirements.
        //

        stridesInfo.setInput(inputEdge(0), inputReqs);
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, outputReqs);
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 1);
        IE_ASSERT(numOutputs() > 0);
        const auto& firstInputPrecision = input(0)->desc().type();
        assertAllInputsOutputsTypes(this, {firstInputPrecision}, {firstInputPrecision});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

void FrontEnd::parseSplit(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(!outputs.empty());

    const auto split = std::dynamic_pointer_cast<ie::SplitLayer>(layer);
    IE_ASSERT(split != nullptr);

    const auto input = inputs[0];

    const auto ieRevAxis = input->desc().numDims() - 1 - checked_cast<int>(split->_axis);
    const auto defPerm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    const auto axis = defPerm.at(checked_cast<size_t>(ieRevAxis));
    _stageBuilder->addSplitStage(model, split->name, split, axis, input, outputs);
}

Stage StageBuilder::addSplitStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        Dim axis,
        const Data& input,
        const DataVector& outputs) {
    std::vector<DimValues> offsets;
    offsets.reserve(outputs.size());
    DimValues curOffset({{axis, 0}});

    auto numSplit = outputs.size();
    auto inputAxisSize = input->desc().dim(axis);
    std::vector<int> outAxisSizes {};
    for (size_t i = 0; i < numSplit; ++i) {
        const int begin = (i + 0) * inputAxisSize / numSplit;
        const int end   = (i + 1) * inputAxisSize / numSplit;
        const int outAxisSize = end - begin;
        
        outAxisSizes.push_back(outAxisSize);
    }
    
    auto outputs_ = outputs;
    auto it = outAxisSizes.begin();
    for (int i = 0; i < outputs_.size() ; ++i) {
        if (outputs_[i] == nullptr) {
            curOffset.set(axis, curOffset[axis] + *it++);
            outputs_.erase(outputs_.begin() + i--);
            continue;
        }
        offsets.emplace_back(curOffset);
        curOffset.set(axis, curOffset[axis] + *it++);
    }

    auto stage = addSplitStage(model, name, layer, std::move(offsets), input, outputs_);

    stage->attrs().set("axis", axis);

    return stage;
}

Stage StageBuilder::addSplitStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        std::vector<vpu::DimValues>&& offsets,
        const Data& input,
        const DataVector& outputs) {
    IE_ASSERT(offsets.size() == outputs.size());

    auto stage = model->addNewStage<SplitStage>(
        name,
        StageType::Split,
        layer,
        {input},
        outputs);

    stage->attrs().set("offsets", std::move(offsets));

    return stage;
}

}  // namespace vpu
