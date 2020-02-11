// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <algorithm>
#include <memory>
#include <set>
#include <string>

namespace vpu {

namespace {

class ReduceStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReduceStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
         auto input0 = inputEdge(0)->input();
         auto input1 = inputEdge(1)->input();
         auto output = outputEdge(0)->output();

         auto in0Desc = input0->desc();
         auto in1Desc = input1->desc();
         auto outDesc = output->desc();

         auto in0Order = DimsOrder::fromNumDims(in0Desc.numDims());
         auto in1Order = DimsOrder::fromNumDims(in1Desc.numDims());
         auto outOrder = DimsOrder::fromNumDims(outDesc.numDims());

         orderInfo.setInput(inputEdge(0), in0Order);
         orderInfo.setInput(inputEdge(1), in1Order);
         orderInfo.setOutput(outputEdge(0), outOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();

        auto in0Desc = input0->desc();
        auto in1Desc = input1->desc();

        IE_ASSERT(input1->usage() == DataUsage::Const);

        size_t ndims = in0Desc.numDims();
        IE_ASSERT(in1Desc.numDims() == 1);
        size_t indicesSize = in1Desc.totalDimSize();
        IE_ASSERT(indicesSize <= ndims);

        const auto oldIndices = input1->content()->get<int32_t>();

        auto newIndicesBlob = ie::make_shared_blob<int32_t>(InferenceEngine::TensorDesc(
            ie::Precision::I32,
            {indicesSize},
            ie::Layout::C));
        newIndicesBlob->allocate();

        auto newIndices = newIndicesBlob->buffer().as<int32_t*>();

        const auto defDimsOrder = DimsOrder::fromNumDims(ndims);
        const auto defPerm = defDimsOrder.toPermutation();
        for (size_t i = 0; i < indicesSize; ++i) {
            auto irIndex = oldIndices[i];
            if (irIndex < 0) {
                // handle negative indices
                irIndex = ndims - irIndex;
            }
            IE_ASSERT(irIndex < ndims);

            const auto irRevIndex = ndims - 1 - irIndex;

            const auto irDim = defPerm[irRevIndex];

            const auto vpuDimInd = in0Desc.dimsOrder().dimInd(irDim);
            newIndices[i] = vpuDimInd;
        }

        std::sort(newIndices, newIndices + indicesSize);

        auto newList = model()->duplicateData(
            input1,
            "",
            DataDesc(),
            ieBlobContent(newIndicesBlob));

        model()->replaceStageInput(inputEdge(1), newList);
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::S32}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto keep_dims = attrs().getOrDefault<int>("keep_dims", 1);

        serializer.append(static_cast<int>(keep_dims));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
         auto input0 = inputEdge(0)->input();
         auto input1 = inputEdge(1)->input();
         auto output = outputEdge(0)->output();

         input0->serializeNewBuffer(serializer);
         output->serializeNewBuffer(serializer);
         input1->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseReduce(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    auto layer = std::dynamic_pointer_cast<ie::ReduceLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stageType = StageType::None;
    if (layer->type == "ReduceAnd") {
        stageType = StageType::ReduceAnd;
    } else if (layer->type == "ReduceMin") {
        stageType = StageType::ReduceMin;
    } else if (layer->type == "ReduceMax") {
        stageType = StageType::ReduceMax;
    } else if (layer->type == "ReduceSum") {
        stageType = StageType::ReduceSum;
    } else if (layer->type == "ReduceMean") {
        stageType = StageType::ReduceMean;
    } else {
        VPU_THROW_EXCEPTION << "Reduce operation: " << layer->type << " is not supported";
    }

    if (inputs.size() != 2) {
        VPU_THROW_EXCEPTION << "Reduce operation: " << layer->type << " requires exactly 2 inputs";
    }

    if (outputs.size() != 1) {
        VPU_THROW_EXCEPTION << "Reduce operation: " << layer->type << " requires exactly 1 output";
    }

    _stageBuilder->addReduceStage(model, layer->name, stageType, layer, layer->keep_dims, inputs, outputs[0]);
}

Stage StageBuilder::addReduceStage(
    const Model& model,
    const std::string& name,
    const StageType reduceType,
    const ie::CNNLayerPtr& layer,
    const bool keep_dims,
    const DataVector& inputs,
    const Data& output) {
    auto stage = model->addNewStage<ReduceStage>(name, reduceType, layer, inputs, {output});

    stage->attrs().set<int>("keep_dims", static_cast<int>(keep_dims));
    return stage;
}

}  // namespace vpu
