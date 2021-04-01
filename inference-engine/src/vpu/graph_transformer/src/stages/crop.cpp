// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace vpu {

namespace {

class CropStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<CropStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        orderInfo.setOutput(outputEdge(0), input(0)->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        const auto inputData  = input(0);
        const auto outputData = output(0);

        const auto dimsOrder = inputData->desc().dimsOrder();

        //
        // Get smallest Dim over which Crop is done.
        //

        auto minCroppedDimInd = dimsOrder.numDims();

        for (const auto& p : inputData->desc().dims()) {
            if (outputData->desc().dim(p.first) != p.second) {
                minCroppedDimInd = std::min(minCroppedDimInd, dimsOrder.dimInd(p.first));
            }
        }

        //
        // Initial StridesRequirement for inputs and output.
        //

        auto inputReqs  = inputData->requiredStrides();
        auto outputReqs = inputReqs;

        //
        // Merge output consumers StridesRequirement.
        //

        for (const auto& consumerEdge : outputData->consumerEdges()) {
            const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

            if (!consumerInfo.hasInput(consumerEdge)) {
                continue;
            }

            const auto& consumerReqs = consumerInfo.getInput(consumerEdge);

            for (int i = 0; i < dimsOrder.numDims(); ++i) {
                if (inputReqs.get(i) == DimStride::Any) {
                    const auto consumerReq = consumerReqs.get(i);
                    if (consumerReq != DimStride::Any) {
                        inputReqs.add(i, consumerReq);
                        outputReqs.add(i, consumerReq);
                    }
                }
            }
        }

        //
        // Remove extra output StridesRequirement.
        //

        for (int i = minCroppedDimInd + 1; i < dimsOrder.numDims(); ++i) {
            outputReqs.remove(i);
        }

        //
        // Return merged StridesRequirements.
        //

        stridesInfo.setInput(inputEdge(0), inputReqs);
        stridesInfo.setOutput(outputEdge(0), outputReqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 1 || numInputs() == 2);
        IE_ASSERT(numOutputs() == 1);
        const auto& firstInputPrecision = input(0)->desc().type();
        assertAllInputsOutputsTypes(this, firstInputPrecision, firstInputPrecision);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_FORMAT("Must never be called");
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_FORMAT("Must never be called");
    }
};

}  // namespace

Stage StageBuilder::addCropStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const DimValues& offset) {
    auto stage = model->addNewStage<CropStage>(
        name,
        StageType::Crop,
        layer,
        {input},
        {output});

    stage->attrs().set<DimValues>("offset", offset);

    return stage;
}

void FrontEnd::parseCrop(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1 || inputs.size() == 2,
                     "Crop: number of inputs must be 1 or 2, actually provided: %u", inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Crop: number of outputs must be 1, actually provided: %u", outputs.size());

    auto axisParam   = layer->GetParamAsInts("axis");
    auto offsetParam = layer->GetParamAsInts("offset");
    VPU_THROW_UNLESS(axisParam.size() == offsetParam.size(),
                     "Crop: sizes of `axis` and `offset` must be equal");

    //
    // Parse offset attribute as DimValues
    //

    DimValues offset;
    const auto ndims = inputs[0]->desc().numDims();
    const auto perm = DimsOrder::fromNumDims(ndims).toPermutation();

    for (int i = 0; i < axisParam.size(); ++i) {
        auto axisVal   = axisParam[i];
        auto offsetVal = offsetParam[i];

        if (axisVal < 0) {
            axisVal += ndims;
        }
        VPU_THROW_UNLESS(axisVal >= 0 && axisVal < ndims,
                         "Layer %s [%s] has invalid axis value. "
                         "Expected: 0 <= axis < %d, Actual: %d",
                         layer->name, layer->type, ndims, axisVal);

        offset.set(perm[ndims - 1 - axisVal], offsetVal);
    }

    VPU_THROW_UNLESS(offset.get(Dim::N, 0) == 0 || model->batchSize() == 1,
                     "Crop: batch cropping is not supported");

    auto stage = model->addNewStage<CropStage>(
            layer->name,
            StageType::Crop,
            layer,
            inputs,
            outputs);

    stage->attrs().set("offset", offset);
}

}  // namespace vpu
