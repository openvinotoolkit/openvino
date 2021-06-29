// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class PSROIPoolingStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PSROIPoolingStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input0 = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        orderInfo.setInput(inputEdge(0), input0->desc().dimsOrder().createMovedDim(Dim::C, 2));
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, 2));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setInput(inputEdge(1), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto group_size = attrs().get<int>("group_size");
        auto output_dim = attrs().get<int>("output_dim");
        auto spatial_scale = attrs().get<float>("spatial_scale");

        serializer.append(static_cast<uint32_t>(group_size));
        serializer.append(static_cast<uint32_t>(output_dim));
        serializer.append(static_cast<float>(spatial_scale));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parsePSROIPooling(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& psROIPooling = ngraph::as_type_ptr<ngraph::op::v0::PSROIPooling>(node);
    IE_ASSERT(psROIPooling != nullptr);
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<PSROIPoolingStage>(psROIPooling->get_name(), StageType::PSROIPooling, psROIPooling, inputs, outputs);
    stage->attrs().set<int>("group_size", psROIPooling->get_group_size());
    stage->attrs().set<int>("output_dim", psROIPooling->get_output_dim());
    stage->attrs().set<float>("spatial_scale", psROIPooling->get_spatial_scale());
}

}  // namespace vpu
