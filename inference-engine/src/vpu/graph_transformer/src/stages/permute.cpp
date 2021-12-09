// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/compile_env.hpp>

#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <limits>

#include <vpu/configuration/options/disable_reorder.hpp>

namespace vpu {

namespace {

const char permutationKey[]          = "permutation";
const char outputOrderKey[]          = "outputOrder";

class PermuteStage : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PermuteStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        DimsOrder outputOrder = input(0)->desc().dimsOrder();
        outputOrder = attrs().getOrDefault(outputOrderKey, outputOrder);
        orderInfo.setOutput(outputEdge(0), outputOrder);
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
        const auto& firstInputPrecision = input(0)->desc().type();
        assertInputsOutputsTypes(this, {{firstInputPrecision}}, {{firstInputPrecision}});
    }

    void finalCheckImpl() const override {
        auto inDimsOrder = input(0)->desc().dimsOrder();
        auto outDimsOrder = output(0)->desc().dimsOrder();
        IE_ASSERT(inDimsOrder.numDims() == outDimsOrder.numDims());
        IE_ASSERT(isOrdersCompatible(inDimsOrder, outDimsOrder));
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& permutation = attrs().get<PermutationDimsMap>(permutationKey);
        PermutationIndexVector indices = permuteMapToVector(permutation,
                                                            input(0)->desc().dimsOrder(),
                                                            output(0)->desc().dimsOrder());
        indices.resize(MAX_DIMS_32, -1);

        for (const auto index : indices) {
            serializer.append(static_cast<uint32_t>(index));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        input(0)->serializeBuffer(serializer);
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parsePermute(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    const auto ieOrder = layer->GetParamAsUInts("order");
    const auto perm = DimsOrder::fromNumDims(checked_cast<int>(ieOrder.size())).toPermutation();

    PermutationDimsMap permutation;
    for (size_t i = 0; i < ieOrder.size(); i++) {
        const auto srcDim = perm[ieOrder.size() - ieOrder[i] - 1];
        const auto dstDim = perm[ieOrder.size() - i - 1];
        permutation.set(dstDim, srcDim);
    }

    _stageBuilder->addPermuteStage(model, layer->name, layer, inputs[0], outputs[0], permutation);
}

Stage StageBuilder::addPermuteStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const PermutationDimsMap& permutation) {
    auto inDimsOrder  = input->desc().dimsOrder();
    auto outDimsOrder = output->desc().dimsOrder();
    IE_ASSERT(isOrdersCompatible(inDimsOrder, outDimsOrder));

    auto stage = model->addNewStage<PermuteStage>(
        name,
        StageType::Permute,
        layer,
        {input},
        {output});
    stage->attrs().set(permutationKey, permutation);
    return stage;
}

Stage StageBuilder::addReorderStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output) {
    const auto* env = CompileEnv::getOrNull();
    VPU_THROW_UNLESS(
        env == nullptr || !env->config.get<DisableReorderOption>(),
        "Tried to add Reorder Stage %v, while DISABLE_REORDER option was set",
        name);

    for (const auto& p : input->desc().dims()) {
        IE_ASSERT(p.second == output->desc().dim(p.first));
    }

    PermutationDimsMap permutationMap;
    for (const auto & dim : output->desc().dimsOrder().toPermutation()) {
        permutationMap.set(dim, dim);
    }

    auto stage = addPermuteStage(model, name, layer, input, output, permutationMap);
    stage->attrs().set(outputOrderKey, output->desc().dimsOrder());
    return stage;
}

}  // namespace vpu
