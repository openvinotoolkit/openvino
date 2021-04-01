// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <cstdio>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

VPU_DECLARE_ENUM(ROIPoolingMethod,
    Max = 0,
    Bilinear = 1
)

namespace {

class ROIPoolingStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ROIPoolingStage>(*this);
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
        auto pooled_w = attrs().get<int>("pooled_w");
        auto pooled_h = attrs().get<int>("pooled_h");
        auto spatial_scale = attrs().get<float>("spatial_scale");
        auto method = attrs().get<ROIPoolingMethod>("method");

        serializer.append(static_cast<uint32_t>(pooled_w));
        serializer.append(static_cast<uint32_t>(pooled_h));
        serializer.append(static_cast<float>(spatial_scale));
        serializer.append(static_cast<uint32_t>(method));
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

void FrontEnd::parseROIPooling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ROIPoolingStage>(layer->name, StageType::ROIPooling, layer, inputs, outputs);

    stage->attrs().set<int>("pooled_w", layer->GetParamAsInt("pooled_w", 7));
    stage->attrs().set<int>("pooled_h", layer->GetParamAsInt("pooled_h", 7));
    stage->attrs().set<float>("spatial_scale", layer->GetParamAsFloat("spatial_scale", 0.0625f));

    auto method = layer->GetParamAsString("method", "max");
    if (cmp(method, "bilinear")) {
        stage->attrs().set("method", ROIPoolingMethod::Bilinear);
    } else {
        stage->attrs().set("method", ROIPoolingMethod::Max);
    }
}

}  // namespace vpu
