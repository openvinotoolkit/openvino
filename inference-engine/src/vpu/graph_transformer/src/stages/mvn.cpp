// Copyright (C) 2018-2020 Intel Corporation
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
        auto eps = attrs().get<float>("eps");

        serializer.append(static_cast<int32_t>(normalize));
        serializer.append(static_cast<int32_t>(across_channels));
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
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    const auto& input = inputs[0];
    const int ndims = input->desc().numDims();
    IE_ASSERT(ndims == 3 || ndims == 4);

    const auto& indices = inputs[1];
    const int indices_size = indices->desc().totalDimSize();
    const auto indices_ptr = indices->content()->get<int>();

    const auto& getDimFromAxis = [](int ndims, int axis_index) -> Dim {
        return DimsOrder::fromNumDims(ndims).toPermutation()[ndims - axis_index - 1];
    };
    std::set<Dim> axes;
    for (int i = 0; i < indices_size; i++) {
        axes.insert(getDimFromAxis(ndims, indices_ptr[i]));
    }

    bool across_channels = false;
    if (!axes.count(Dim::N) && axes.count(Dim::H) && axes.count(Dim::W)) {
        across_channels = axes.count(Dim::C) != 0;
    } else {
        VPU_THROW_FORMAT("Unsupported combination of indices in layer \"%s\". "
                         "Only across channel and full batch supported.", layer->name);
    }

    const auto norm_variance = layer->GetParamAsBool("normalize_variance");
    const auto eps = layer->GetParamAsFloat("eps");
    const auto eps_mode = layer->GetParamAsString("eps_mode", "outside_sqrt");
    VPU_THROW_UNLESS(eps_mode == "outside_sqrt", "Only eps_mode == \"outside_sqrt\" supported.");

    auto stage = model->addNewStage<MVNStage>(layer->name, StageType::MVN, layer, inputs, outputs);
    stage->attrs().set<int>("normalize", norm_variance);
    stage->attrs().set<int>("across_channels", across_channels);
    stage->attrs().set<float>("eps", eps);
}

}  // namespace vpu
