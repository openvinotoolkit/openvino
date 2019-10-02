// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <set>

namespace vpu {

static TopKMode getMode(const std::shared_ptr<ie::TopKLayer> layer) {
    const auto& mode = layer->mode;
    if (mode == "max")
        return TopKMode::Max;
    if (mode == "min")
        return TopKMode::Min;
    VPU_THROW_EXCEPTION << layer->name << " TopK can take only 'max' or 'min' for mode, but actually it has: " << mode;
}

static TopKSort getSort(const std::shared_ptr<ie::TopKLayer> layer) {
    const auto& sort = layer->sort;
    if (sort == "none")
        return TopKSort::None;
    if (sort == "value")
        return TopKSort::Value;
    if (sort == "index")
        return TopKSort::Index;
    VPU_THROW_EXCEPTION << layer->name << " TopK can take only 'value', 'index' or 'none' for sort, but actually it has: " << sort;
}

namespace {

class TopKStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<TopKStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto inputValues = input(0);

        auto outputOrder = inputValues->desc().dimsOrder();

        orderInfo.setOutput(outputEdge(0), outputOrder);
        orderInfo.setOutput(outputEdge(1), outputOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& /*stridesInfo*/) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
            {{DataType::FP16}, {DataType::S32}},
            {{DataType::FP16}, {DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto inputValues = input(0);

        auto axis = attrs().get<Dim>("axis");
        auto axisInd = inputValues->desc().dimsOrder().dimInd(axis);

        auto mode = attrs().get<TopKMode>("mode");
        auto sort = attrs().get<TopKSort>("sort");

        serializer.append(static_cast<int32_t>(axisInd));
        serializer.append(static_cast<int32_t>(mode));
        serializer.append(static_cast<int32_t>(sort));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto inputValues = input(0);
        auto inputK = input(1);
        auto outputValues = output(0);
        auto outputIndices = output(1);

        inputValues->serializeNewBuffer(serializer);
        outputValues->serializeNewBuffer(serializer);
        inputK->serializeNewBuffer(serializer);
        outputIndices->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseTopK(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    auto layer = std::dynamic_pointer_cast<ie::TopKLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 2);

    auto inputValues = inputs[0];
    auto inputK = inputs[1];
    auto outputValues = outputs[0];
    auto outputIndices = outputs[1];

    const auto numDims = inputValues->desc().numDims();

    IE_ASSERT(inputK->desc().numDims() == 1);
    IE_ASSERT(outputValues->desc().numDims() == numDims);
    IE_ASSERT(outputIndices->desc().numDims() == numDims);

    IE_ASSERT(layer->axis < numDims);

    auto perm = DimsOrder::fromNumDims(numDims).toPermutation();
    auto axis = perm[numDims - 1 - layer->axis];

    TopKMode mode = getMode(layer);
    TopKSort sort = getSort(layer);

    auto stage = model->addNewStage<TopKStage>(
        layer->name,
        StageType::TopK,
        layer,
        inputs,
        outputs);

    stage->attrs().set<Dim>("axis", axis);
    stage->attrs().set<TopKMode>("mode", mode);
    stage->attrs().set<TopKSort>("sort", sort);
}

}  // namespace vpu
