// Copyright (C) 2018-2020 Intel Corporation
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

        const auto outs = attrs().get<TopKOutputs>("outputs");

        orderInfo.setOutput(outputEdge(0), outputOrder);

        if (outs == TopKOutputs::All)
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
        const auto outs = attrs().get<TopKOutputs>("outputs");

        const DataTypesRequirement expectedOutputsTypes =
                outs == TopKOutputs::All        ? DataTypesRequirement{{DataType::FP16}, {DataType::S32}} :
                outs == TopKOutputs::ValueOnly  ? DataTypesRequirement{{DataType::FP16}}                  :
                outs == TopKOutputs::IndexOnly  ? DataTypesRequirement{{DataType::S32}}                   : DataTypesRequirement{};

        assertInputsOutputsTypes(this,
            {{DataType::FP16}, {DataType::S32}},
            expectedOutputsTypes);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto inputValues = input(0);

        const auto axis = attrs().get<Dim>("axis");
        const auto axisInd = inputValues->desc().dimsOrder().dimInd(axis);

        const auto mode = attrs().get<TopKMode>("mode");
        const auto sort = attrs().get<TopKSort>("sort");
        const auto outs = attrs().get<TopKOutputs>("outputs");

        // @note: int32_t instead of bool because firmware require overall alignment of parameters
        const int32_t valuesPresent  = (outs == TopKOutputs::All || outs == TopKOutputs::ValueOnly);
        const int32_t indicesPresent = (outs == TopKOutputs::All || outs == TopKOutputs::IndexOnly);

        serializer.append(static_cast<int32_t>(axisInd));
        serializer.append(static_cast<int32_t>(mode));
        serializer.append(static_cast<int32_t>(sort));

        serializer.append(valuesPresent);
        serializer.append(indicesPresent);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto inputValues = input(0);
        auto inputK = input(1);

        inputValues->serializeBuffer(serializer);
        inputK->serializeBuffer(serializer);

        const auto outs = attrs().get<TopKOutputs>("outputs");

        if (outs == TopKOutputs::All || outs == TopKOutputs::ValueOnly)
            output(0)->serializeBuffer(serializer);

        if (outs == TopKOutputs::All || outs == TopKOutputs::IndexOnly)
            output(outs == TopKOutputs::IndexOnly ? 0 : 1)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseTopK(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
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
    IE_ASSERT(outputValues || outputIndices);
    IE_ASSERT(!outputValues || outputValues->desc().numDims() == numDims);
    IE_ASSERT(!outputIndices || outputIndices->desc().numDims() == numDims);

    IE_ASSERT(layer->axis < numDims);

    auto perm = DimsOrder::fromNumDims(numDims).toPermutation();
    auto axis = perm[numDims - 1 - layer->axis];

    const TopKMode mode = getMode(layer);
    const TopKSort sort = getSort(layer);
    TopKOutputs outputsMode = TopKOutputs::All;
    DataVector realOutputs = outputs;
    if (!outputValues) {
        outputsMode = TopKOutputs::IndexOnly;
        realOutputs = {outputIndices};
    }
    if (!outputIndices) {
        outputsMode = TopKOutputs::ValueOnly;
        realOutputs = {outputValues};
    }

    auto stage = model->addNewStage<TopKStage>(layer->name,
                                               StageType::TopK,
                                               layer, inputs, realOutputs);

    stage->attrs().set<Dim>("axis", axis);
    stage->attrs().set<TopKMode>("mode", mode);
    stage->attrs().set<TopKSort>("sort", sort);
    stage->attrs().set<TopKOutputs>("outputs", outputsMode);
}

}  // namespace vpu
