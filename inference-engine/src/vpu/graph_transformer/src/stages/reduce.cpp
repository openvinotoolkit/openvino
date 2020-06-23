// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/model/data_desc.hpp>

#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <algorithm>
#include <memory>
#include <set>
#include <vector>
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
         orderInfo.setInput(inputEdge(0), input(0)->desc().dimsOrder());
         orderInfo.setInput(inputEdge(1), input(1)->desc().dimsOrder());
         orderInfo.setOutput(outputEdge(0), output(0)->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
        auto reductionAxes = input(1);
        auto in0Desc = input(0)->desc();
        auto in1Desc = reductionAxes->desc();

        VPU_THROW_UNLESS(reductionAxes->usage() == DataUsage::Const,
                        "Stage {} of type {} expects input with index {} ({}) to be {}, but it is {}",
                        name(), type(), 1, reductionAxes->name(), DataUsage::Const, reductionAxes->usage());
        size_t ndims = in0Desc.numDims();
        VPU_THROW_UNLESS(in1Desc.numDims() == 1,
                        "Stage {} of type {} expects input with index {} ({}) to have dimensions number is {}, but it is {}",
                        name(), type(), 1, reductionAxes->name(), 1, in1Desc.numDims());
        size_t indicesSize = in1Desc.totalDimSize();
        VPU_THROW_UNLESS(indicesSize <= ndims,
                        "Stage {} of type {} expects input with index {} ({}) to have total size not greater than dimensions ",
                        "number of input with index {} ({}), but it is {} > {}",
                        name(), type(), 1, reductionAxes->name(), 0, input(0)->name(), indicesSize, ndims);

        const auto oldIndices = reductionAxes->content()->get<int32_t>();

        auto newIndicesBlob = ie::make_shared_blob<int32_t>(InferenceEngine::TensorDesc(
            ie::Precision::I32,
            {indicesSize},
            ie::Layout::C));
        newIndicesBlob->allocate();

        auto newIndices = newIndicesBlob->buffer().as<int32_t*>();

        const auto defPerm = DimsOrder::fromNumDims(ndims).toPermutation();
        const auto dimsOrder = in0Desc.dimsOrder();
        for (size_t i = 0; i < indicesSize; ++i) {
            auto irIndex = oldIndices[i];
            if (irIndex < 0) {
                // handle negative indices
                irIndex = ndims - std::abs(irIndex);
            }
            VPU_THROW_UNLESS(irIndex < ndims,
                            "Stage {} of type {} expects input with index {} ({}) include values less than ",
                            "dimensions number of input with index {} ({}), but it is {} >= {}",
                             name(), type(), 1, reductionAxes->name(), 0, input(0)->name(), irIndex, ndims);

            const auto reducedDim = defPerm[ndims - 1 - irIndex];
            newIndices[i] = dimsOrder.dimInd(reducedDim);
        }
        std::sort(newIndices, newIndices + indicesSize);

        auto newList = model()->duplicateData(
            reductionAxes,
            "",
            DataDesc(),
            ieBlobContent(newIndicesBlob, DataType::S32));

        model()->replaceStageInput(inputEdge(1), newList);
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        VPU_THROW_UNLESS(input(0)->desc().type() == output(0)->desc().type(),
                         "Stage {} of type {} expects that data types of input with index {} ({}) ",
                         "and output with index {} ({}) are the same, but it is {} and {}",
                         name(), type(), 0, input(0)->name(), 0, output(0)->name(), input(0)->desc().type(), output(0)->desc().type());
        assertInputsOutputsTypes(this,
                                 {{DataType::FP16, DataType::S32}, {DataType::S32}},
                                 {{DataType::FP16, DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto keep_dims = attrs().getOrDefault<int>("keep_dims", 1);

        serializer.append(static_cast<int>(keep_dims));
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

void FrontEnd::parseReduce(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    auto layer = std::dynamic_pointer_cast<ie::ReduceLayer>(_layer);
    VPU_THROW_UNLESS(layer != nullptr, "parseReduce expects valid ReduceLayer, actually got nullptr");

    VPU_THROW_UNLESS(inputs.size() == 2,
                     "Layer {} of type {} expects {} inputs, but provided {}",
                     layer->name, layer->type, 2, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Layer {} of type {} expects {} output, but provided {}",
                     layer->name, layer->type, 1, outputs.size());

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
