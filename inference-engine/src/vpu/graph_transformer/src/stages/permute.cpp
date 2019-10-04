// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <utility>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

namespace vpu {

namespace {

class PermuteStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PermuteStage>(*this);
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
        orderInfo.setOutput(outputEdge(0), input(0)->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>&) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>&) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& permutation = attrs().get<DimValues_<Dim>>("permutation");

        for (auto dstDim : output(0)->desc().dimsOrder().toPermutation()) {
            const auto srcDim = permutation[dstDim];
            const auto srcDimInd = input(0)->desc().dimsOrder().dimInd(srcDim);
            serializer.append(static_cast<uint32_t>(srcDimInd));
        }

        for (int i = output(0)->desc().numDims(); i < MAX_DIMS_32; i++) {
            serializer.append(static_cast<uint32_t>(-1));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parsePermute(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    const auto ieOrder = layer->GetParamAsUInts("order");
    const auto perm = DimsOrder::fromNumDims(checked_cast<int>(ieOrder.size())).toPermutation();

    DimValues_<Dim> permutation;
    for (size_t i = 0; i < ieOrder.size(); i++) {
        const auto srcDim = perm[ieOrder.size() - ieOrder[i] - 1];
        const auto dstDim = perm[ieOrder.size() - i - 1];
        permutation.set(dstDim, srcDim);
    }

    _stageBuilder->addPermuteStage(model, layer->name, layer, inputs[0], outputs[0], permutation);
}

Stage StageBuilder::addPermuteStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const DimValues_<Dim>& permutation) {
    auto stage = model->addNewStage<PermuteStage>(
        name,
        StageType::Permute,
        layer,
        {input},
        {output});
    stage->attrs().set("permutation", permutation);

    return stage;
}

}  // namespace vpu
