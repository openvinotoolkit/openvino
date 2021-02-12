// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <algorithm>

namespace vpu {

namespace {

class CTCGreedyDecoderSeqLenStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CTCGreedyDecoderSeqLenStage>(*this);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>&) override {
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output0 = outputEdge(0)->output();
        auto output1 = outputEdge(1)->output();

        orderInfo.setInput(inputEdge(0), DimsOrder::fromNumDims(input0->desc().numDims()));
        orderInfo.setInput(inputEdge(1), DimsOrder::fromNumDims(input1->desc().numDims()));
        orderInfo.setOutput(outputEdge(0), DimsOrder::fromNumDims(output0->desc().numDims()));
        orderInfo.setOutput(outputEdge(1), DimsOrder::fromNumDims(output1->desc().numDims()));

        if (numInputs() == 3) {
            auto input2 = inputEdge(2)->input();
            orderInfo.setInput(inputEdge(2), DimsOrder::fromNumDims(input2->desc().numDims()));
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setInput(inputEdge(1), StridesRequirement::compact());
        if (numInputs() == 3) {
            stridesInfo.setInput(inputEdge(2), StridesRequirement::compact());
        }
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(1), StridesRequirement::compact());
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void initialCheckImpl() const override {
        VPU_THROW_UNLESS(numInputs() == 2 || numInputs() == 3,
                         "{} stage with name {} must have 2 or 3 inputs, actually "
                         "provided {} inputs",
                         type(), name(), numInputs());
        VPU_THROW_UNLESS(numOutputs() == 2,
                         "{} stage with name {} must have 2 output, actually "
                         "provided {} outputs",
                         type(), name(), numOutputs());

        if (numInputs() == 2) {
            assertInputsOutputsTypes(
                this, {{DataType::FP16}, {DataType::S32}},
                {{DataType::S32}, {DataType::S32}});
        } else {
            assertInputsOutputsTypes(
                this, {{DataType::FP16}, {DataType::S32}, {DataType::S32}},
                {{DataType::S32}, {DataType::S32}});
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto mergeRepeated = attrs().get<bool>("mergeRepeated");
        const auto blankIndex = attrs().get<int32_t>("blankIndex");

        serializer.append(static_cast<uint32_t>(mergeRepeated));
        serializer.append(static_cast<uint32_t>(blankIndex));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output0 = outputEdge(0)->output();
        auto output1 = outputEdge(1)->output();

        input0->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
        output0->serializeBuffer(serializer);
        output1->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addCTCGreedyDecoderSeqLenStage(const Model& model,
                                                   const std::string& name,
                                                   const ie::CNNLayerPtr& layer,
                                                   const DataVector& inputs,
                                                   const DataVector& outputs,
                                                   bool mergeRepeated,
                                                   int32_t blankIndex) {
    auto stage = model->addNewStage<CTCGreedyDecoderSeqLenStage>(name,
                                                                 StageType::CTCGreedyDecoderSeqLen,
                                                                 layer, inputs, outputs);
    stage->attrs().set<bool>("mergeRepeated", mergeRepeated);
    stage->attrs().set<int32_t>("blankIndex", blankIndex);

    return stage;
}

void FrontEnd::parseCTCGreedyDecoderSeqLen(const Model& model, const ie::CNNLayerPtr& layer,
                                           const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(layer, "CNNLayer pointer is null.");
    VPU_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                     "{} layer with name {} must have 2 or 3 inputs, actually "
                     "provided {} inputs",
                     layer->type, layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 2,
                     "{} layer with name {} must have 2 outputs, actually "
                     "provided {} outputs",
                     layer->type, layer->name, outputs.size());


    const auto mergeRepeated = layer->GetParamAsBool("merge_repeated");
    const auto blankIndex = [&] {
        if (inputs.size() == 3) {
            VPU_THROW_UNLESS(inputs[2]->usage() == DataUsage::Const,
                             "Only constant axis is supported, but got {} data object",
                             inputs[2]->usage());
            VPU_THROW_UNLESS(inputs[2]->desc().totalDimSize() == 1,
                             "Only single value blankIndex is supported, got {} elements",
                             inputs[2]->desc().totalDimSize());

            return *inputs[2]->content()->get<int32_t>();
        }

        const auto classes = inputs[0]->desc().dim(Dim::W);
        return classes - 1;
    }();

    const auto toUpper = [](const std::string& str) {
        std::string result;
        result.reserve(str.size());
        std::transform(begin(str), end(str), std::back_inserter(result),
                       [](char c) {return std::toupper(c);} );
        return result;
    };

    const auto classesIndexType = toUpper(layer->GetParamAsString("classes_index_type"));
    const auto sequenceLengthType = toUpper(layer->GetParamAsString("sequence_length_type"));

    VPU_THROW_UNLESS(classesIndexType == "I32", "classes_index_type == %s. Only I32 is supported",
                     classesIndexType);

    VPU_THROW_UNLESS(sequenceLengthType == "I32", "sequence_length_type == %s. Only I32 is supported",
                     sequenceLengthType);

    _stageBuilder->addCTCGreedyDecoderSeqLenStage(model, layer->name, layer,
                                                  inputs, outputs, mergeRepeated, blankIndex);
}

}  // namespace vpu
