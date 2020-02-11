// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <set>
#include <string>
#include <utility>

namespace vpu {

namespace {

class ConvertStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConvertStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto& input = inputEdge(0)->input();
        const auto& output = outputEdge(0)->output();

        if (input->usage() == DataUsage::Output) {
            const auto& outDimsOrder = output->desc().dimsOrder();

            // HCW is not supported
            IE_ASSERT(outDimsOrder.dimInd(Dim::C) != 1);

            orderInfo.setInput(inputEdge(0), outDimsOrder);
        } else {
            const auto& inDimsOrder = input->desc().dimsOrder();

            // HCW is not supported
            IE_ASSERT(inDimsOrder.dimInd(Dim::C) != 1);

            orderInfo.setOutput(outputEdge(0), inDimsOrder);
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        const auto& input = inputEdge(0)->input();
        const auto& inDimsOrder = input->desc().dimsOrder();

        StridesRequirement reqs;
        if (input->desc().dim(Dim::N, 1) > 1) {
            // To merge batch into previous dimension.
            reqs.add(inDimsOrder.dimInd(Dim::N), DimStride::Compact);
        }

        stridesInfo.setInput(inputEdge(0), reqs);
        stridesInfo.setOutput(outputEdge(0), reqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        // Convert will support batch by merging it with previous dimension.
    }

    void initialCheckImpl() const override {
        const auto expectedTypes = std::set<std::pair<DataType, DataType>>{
            {DataType::U8, DataType::FP16},
            {DataType::FP16, DataType::FP32},
            {DataType::FP32, DataType::FP16},
            {DataType::S32, DataType::FP16},
            {DataType::FP16, DataType::S32},
        };

        const auto inType = inputEdge(0)->input()->desc().type();
        const auto outType = outputEdge(0)->output()->desc().type();

        const auto typePair = std::make_pair(inType, outType);
        const auto match = expectedTypes.find(typePair);
        IE_ASSERT(match != expectedTypes.end()) << "Unsupported data type conversion";

        assertInputsOutputsTypes(this, {{inType}}, {{outType}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto scale = attrs().getOrDefault<float>("scale", 1.f);
        const auto bias = attrs().getOrDefault<float>("bias", 0.f);
        const auto convertFromDetOutput = attrs().getOrDefault<bool>("convertFromDetOutput", false);
        const auto haveBatch = attrs().getOrDefault<bool>("haveBatch", true);

        serializer.append(static_cast<float>(scale));
        serializer.append(static_cast<float>(bias));
        serializer.append(static_cast<int32_t>(convertFromDetOutput));
        serializer.append(static_cast<int32_t>(haveBatch));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto& input = inputEdge(0)->input();
        const auto& output = outputEdge(0)->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::createConvertStage(
        const Model& model,
        const std::string& name,
        const Data& input,
        const Data& output,
        float scale,
        float bias) {
    auto stage = model->addNewStage<ConvertStage>(
        name,
        StageType::Convert,
        nullptr,
        {input},
        {output});

    stage->attrs().set("scale", scale);
    stage->attrs().set("bias", bias);

    return stage;
}

void FrontEnd::parseConvert(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ConvertStage>(layer->name, StageType::Convert, layer, inputs, outputs);
    stage->attrs().set("scale", 1.f);
    stage->attrs().set("bias", 0.f);
}

}  // namespace vpu
