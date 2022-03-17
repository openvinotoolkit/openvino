// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <precision_utils.h>
#include <memory>
#include <set>

namespace vpu {
namespace {
class OneHot final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<OneHot>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();
        orderInfo.setInput(inputEdge(0), DimsOrder::fromNumDims(input->desc().numDims()));
        orderInfo.setOutput(outputEdge(0), DimsOrder::fromNumDims(output->desc().numDims()));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::S32}},
                                 {{DataType::FP16}});
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto axis = attrs().get<int>("axis");
        auto depth = attrs().get<unsigned int>("depth");
        auto on_value = attrs().get<float>("on_value");
        auto off_value = attrs().get<float>("off_value");

        serializer.append(static_cast<int32_t>(axis));
        serializer.append(static_cast<uint32_t>(depth));
        serializer.append(on_value);
        serializer.append(off_value);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(inputEdges().size() == 1);
        IE_ASSERT(outputEdges().size() == 1);

        auto input = inputEdges()[0]->input();
        auto output = outputEdges()[0]->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseOneHot(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto oneHot = std::dynamic_pointer_cast<ie::OneHotLayer>(layer);
    IE_ASSERT(oneHot != nullptr);

    auto axis = oneHot->axis == -1 ? 0 : inputs[0]->desc().numDims() - oneHot->axis;

    auto stage = model->addNewStage<OneHot>(layer->name, StageType::OneHot, layer, inputs, outputs);

    stage->attrs().set<int>("axis", axis);
    stage->attrs().set<unsigned int>("depth", oneHot->depth);
    stage->attrs().set<float>("on_value", oneHot->on_value);
    stage->attrs().set<float>("off_value", oneHot->off_value);
}

}  // namespace vpu
