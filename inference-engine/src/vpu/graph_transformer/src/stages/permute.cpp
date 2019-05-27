// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <utility>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

template <class Cont1, class Cont2>
std::vector<typename Cont1::value_type> permuteArray(const Cont1& src, const Cont2& permutation) {
    std::vector<typename Cont1::value_type> out(permutation.size());

    for (int i = 0; i < out.size(); i++) {
        auto newInd = static_cast<int>(permutation[i]);

        IE_ASSERT(newInd >= 0);
        IE_ASSERT(newInd < src.size());

        out[i] = src[newInd];
    }

    return out;
}

class PermuteStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PermuteStage>(*this);
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
        return DataMap<BatchSupport>();
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        const auto& order = attrs().get<std::vector<int>>("order");

        auto perm = input->desc().dimsOrder().toPermutation();
        auto ind = input->desc().dimsOrder().toIndices();

        auto dimPerm = permuteArray(order, perm);
        auto memoryOrderPerm = permuteArray(ind.toVector(-1), dimPerm);

        int i = 0;
        for (i = 0; i < memoryOrderPerm.size(); i++) {
            serializer.append(static_cast<uint32_t>(memoryOrderPerm[i]));
        }
        for (; i < MAX_DIMS_32; i++) {
            serializer.append(static_cast<uint32_t>(-1));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

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

    auto ieOrder = layer->GetParamAsInts("order");

    auto maxIeOrder = *std::max_element(ieOrder.begin(), ieOrder.end());

    std::vector<int> vpuOrder(MAX_DIMS_64, -1);
    for (size_t i = 0; i < ieOrder.size(); i++) {
        vpuOrder[i] = maxIeOrder - ieOrder[ieOrder.size() - 1 - i];
    }

    auto input = inputs[0];
    auto output = outputs[0];

    auto stage = model->addNewStage<PermuteStage>(
        layer->name,
        StageType::Permute,
        layer,
        inputs,
        outputs);

    stage->attrs().set<std::vector<int>>("order", vpuOrder);
}

}  // namespace vpu
