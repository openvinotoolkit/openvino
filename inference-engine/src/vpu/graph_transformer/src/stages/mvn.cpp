// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>
#include <precision_utils.h>

namespace vpu {

namespace {

class MVNStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<MVNStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::S32}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto normalize = attrs().get<int>("normalize");
        auto across_channels = attrs().get<int>("across_channels");
        auto across_width = attrs().get<int>("across_width");
        auto eps = attrs().get<float>("eps");

        serializer.append(static_cast<int32_t>(normalize));
        serializer.append(static_cast<int32_t>(across_channels));
        serializer.append(static_cast<int32_t>(across_width));
        serializer.append(static_cast<float>(eps));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseMVN(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 2, "%d inputs provided to %s layer, but 2 expected.",
                                            inputs.size(), layer->name);
    VPU_THROW_UNLESS(outputs.size() == 1, "%d outputs provided to %s layer, but 1 expected.",
                                            outputs.size(), layer->name);

    const auto& input = inputs[0];
    const auto ndims = input->desc().numDims();
    VPU_THROW_UNLESS(ndims == 3 || ndims == 4, "%d input rank provided to %s layer, but only 3D and 4D supported.",
                                            ndims, layer->name);

    const auto& indices = inputs[1];
    const auto indicesSize = indices->desc().totalDimSize();
    const auto indicesPtr = indices->content()->get<int>();

    const auto& getDimFromAxis = [](int ndims, int axisIndex) -> Dim {
        return DimsOrder::fromNumDims(ndims).toPermutation()[ndims - axisIndex - 1];
    };
    DimSet axes;
    for (int i = 0; i < indicesSize; i++) {
        axes.insert(getDimFromAxis(ndims, indicesPtr[i]));
    }
    const auto width = axes.count(Dim::W);

    VPU_THROW_UNLESS(!axes.count(Dim::N) && width,
                     "Unsupported combination of indices in layer \"%s\". "
                     "Only across channel and full batch supported.", layer->name);
    const auto acrossChannels = axes.count(Dim::C) != 0;
    const auto acrossWidth = width == 1 && axes.count(Dim::H) == 0;

    const auto normVariance = layer->GetParamAsBool("normalize_variance");
    const auto eps = layer->GetParamAsFloat("eps");
    const auto epsMode = layer->GetParamAsString("eps_mode", "outside_sqrt");
    VPU_THROW_UNLESS(epsMode == "outside_sqrt",
                     "eps_mode == %s provided to %s layer, but only eps_mode == \"outside_sqrt\" supported.",
                     epsMode, layer->name);

    auto stage = model->addNewStage<MVNStage>(layer->name, StageType::MVN, layer, inputs, outputs);
    stage->attrs().set<int>("normalize", normVariance);
    stage->attrs().set<int>("across_channels", acrossChannels);
    stage->attrs().set<int>("across_width", acrossWidth);
    stage->attrs().set<float>("eps", eps);
}

}  // namespace vpu
