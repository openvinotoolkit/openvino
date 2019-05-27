// Copyright (C) 2018-2019 Intel Corporation
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
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ROIPoolingStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>&,
            ScalePropagationStep) override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        out[input0] = 1.0f;
        out[input1] = 1.0f;
        out[output] = 1.0f;

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        DataMap<DimsOrder> out;

        out[input0] = input0->desc().dimsOrder().createMovedDim(Dim::C, 2);
        out[output] = output->desc().dimsOrder().createMovedDim(Dim::C, 2);

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        DataMap<StridesRequirement> out;

        out[input0] = StridesRequirement::compact();
        out[input1] = StridesRequirement::compact();
        out[output] = StridesRequirement::compact();

        return out;
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
    }

    void finalCheckImpl() const override {
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
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
        input1->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseROIPooling(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ROIPoolingStage>(
        layer->name,
        StageType::ROIPooling,
        layer,
        inputs,
        outputs);

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
