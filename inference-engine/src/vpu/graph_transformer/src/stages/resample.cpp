// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

namespace vpu {

VPU_DECLARE_ENUM(ResampleType,
    Nearest  = 0,  // Currently this is only one supported
    Linear = 1,
    Cubic = 2
)

namespace {

class ResampleStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ResampleStage>(*this);
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
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto antialias = attrs().get<bool>("antialias");
        auto factor = attrs().get<float>("factor");
        auto sampleType = attrs().get<ResampleType>("type");

        serializer.append(static_cast<int32_t>(antialias));
        serializer.append(static_cast<float>(factor));
        serializer.append(static_cast<uint32_t>(sampleType));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseResample(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    ie::details::CaselessEq<std::string> cmp;

    auto stage = model->addNewStage<ResampleStage>(layer->name, StageType::Resample, layer, inputs, outputs);

    stage->attrs().set<bool>("antialias", layer->GetParamAsInt("antialias", 0));
    stage->attrs().set<float>("factor", layer->GetParamAsInt("factor", -1.0f));

    auto method = layer->GetParamAsString("type", "caffe.ResampleParameter.NEAREST");
    if (cmp(method, "caffe.ResampleParameter.NEAREST")) {
        stage->attrs().set<ResampleType>("type", ResampleType::Nearest);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " supports only caffe.ResampleParameter.NEAREST resample type";
    }
}

}  // namespace vpu
