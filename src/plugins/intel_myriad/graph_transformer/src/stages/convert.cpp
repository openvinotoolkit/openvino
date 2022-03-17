// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <set>
#include <string>
#include <utility>

namespace vpu {

namespace {

using DataTypeConversionPair = std::pair<DataType, DataType>;
using SupportedConversionSet = std::set<DataTypeConversionPair>;

class ConvertStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    static const SupportedConversionSet expectedTypes;

    StagePtr cloneImpl() const override {
        return std::make_shared<ConvertStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto& input = inputEdge(0)->input();
        const auto& output = outputEdge(0)->output();

        if (output->usage() == DataUsage::Output) {
            orderInfo.setInput(inputEdge(0), output->desc().dimsOrder());
        } else {
            orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        // TODO: #-26090 Convert kernel support only inner stride for now.
        StridesRequirement reqs = StridesRequirement::compact();
        reqs.remove(1);

        stridesInfo.setInput(inputEdge(0), reqs);
        stridesInfo.setOutput(outputEdge(0), reqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        // Convert will support batch by merging it with previous dimension.
    }

    void finalCheckImpl() const override {
        const auto inType = inputEdge(0)->input()->desc().type();
        const auto outType = outputEdge(0)->output()->desc().type();

        VPU_INTERNAL_CHECK(inType != outType,
                           "Final check for stage %v with type %v has failed: "
                           "Conversion to the same data type (%v -> %v) must be already eliminated",
                           name(), type(), inType, outType);

        const auto typePair = std::make_pair(inType, outType);
        VPU_INTERNAL_CHECK(expectedTypes.count(typePair),
                           "Final check for stage %v with type %v has failed: "
                           "Conversion from %v to %v is unsupported",
                           name(), type(), inType, outType);

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

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

const SupportedConversionSet ConvertStage::expectedTypes = {
        {DataType::U8, DataType::FP16},
        {DataType::FP16, DataType::FP32},
        {DataType::FP32, DataType::FP16},
        {DataType::S32, DataType::FP16},
        {DataType::FP16, DataType::S32},
        {DataType::S32, DataType::U8},
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

void FrontEnd::parseConvert(
        const Model &model,
        const ie::CNNLayerPtr &layer,
        const DataVector &inputs,
        const DataVector &outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Convert stage with name %s has invalid number of inputs: expected 1, "
                     "actually provided %u", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Convert stage with name %s has invalid number of outputs: expected 1, "
                     "actually provided %u", layer->name, outputs.size());

    auto stage = model->addNewStage<ConvertStage>(
            layer->name,
            StageType::Convert,
            layer, inputs,
            outputs);

    stage->attrs().set("scale", 1.f);
    stage->attrs().set("bias", 0.f);
}

}  // namespace vpu
