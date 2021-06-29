// Copyright (C) 2018-2021 Intel Corporation
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
                irIndex = static_cast<int>(ndims - std::abs(irIndex));
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
                         "Stage {} of type {} expects that data types of input with index {} ({}) "
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

void FrontEnd::parseReduceImpl(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs, vpu::StageType stageType, bool keepDims) const {
    VPU_THROW_UNLESS(node != nullptr,
                     "parseReduce expects valid NodePtr, got nullptr");
    VPU_THROW_UNLESS(node != nullptr,
                     "Layer {} of type {} cannot be casted to ie::ReduceLayer",
                     node->get_friendly_name(), node->get_type_name());
    VPU_THROW_UNLESS(inputs.size() == 2,
                     "Layer {} of type {} expects {} inputs, but provided {}",
                     node->get_friendly_name(), node->get_type_name(), 2, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Layer {} of type {} expects {} output, but provided {}",
                     node->get_friendly_name(), node->get_type_name(), 1, outputs.size());

    if (inputs.size() != 2) {
        VPU_THROW_EXCEPTION << "Reduce operation: " << node->get_type_name() << " requires exactly 2 inputs";
    }

    if (outputs.size() != 1) {
        VPU_THROW_EXCEPTION << "Reduce operation: " << node->get_type_name() << " requires exactly 1 output";
    }

    _stageBuilder->addReduceStage(model, node->get_friendly_name(), stageType, node, keepDims, inputs, outputs[0]);
}

void FrontEnd::parseReduceSum(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto reduce = ngraph::as_type_ptr<ngraph::opset4::ReduceSum>(node);
    VPU_THROW_UNLESS(reduce != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());

    _stageBuilder->addReduceStage(model, node->get_friendly_name(), vpu::StageType::ReduceSum, node, reduce->get_keep_dims(), inputs, outputs[0]);
}
void FrontEnd::parseReduceMax(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto reduce = ngraph::as_type_ptr<ngraph::opset4::ReduceMax>(node);
    VPU_THROW_UNLESS(reduce != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());

    _stageBuilder->addReduceStage(model, node->get_friendly_name(), vpu::StageType::ReduceMax, node, reduce->get_keep_dims(), inputs, outputs[0]);
}
void FrontEnd::parseReduceMin(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto reduce = ngraph::as_type_ptr<ngraph::opset4::ReduceMin>(node);
    VPU_THROW_UNLESS(reduce != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());

    _stageBuilder->addReduceStage(model, node->get_friendly_name(), vpu::StageType::ReduceMin, node, reduce->get_keep_dims(), inputs, outputs[0]);
}
void FrontEnd::parseReduceAnd(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto reduce = ngraph::as_type_ptr<ngraph::opset4::ReduceLogicalAnd>(node);
    VPU_THROW_UNLESS(reduce != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());

    _stageBuilder->addReduceStage(model, node->get_friendly_name(), vpu::StageType::ReduceAnd, node, reduce->get_keep_dims(), inputs, outputs[0]);
}
void FrontEnd::parseReduceMean(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto reduce = ngraph::as_type_ptr<ngraph::opset4::ReduceMean>(node);
    VPU_THROW_UNLESS(reduce != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());

    _stageBuilder->addReduceStage(model, node->get_friendly_name(), vpu::StageType::ReduceMean, node, reduce->get_keep_dims(), inputs, outputs[0]);
}

Stage StageBuilder::addReduceStage(
    const Model& model,
    const std::string& name,
    const StageType reduceType,
    const NodePtr& node,
    const bool keep_dims,
    const DataVector& inputs,
    const Data& output) {
    auto stage = model->addNewStage<ReduceStage>(name, reduceType, node, inputs, {output});

    stage->attrs().set<int>("keep_dims", static_cast<int>(keep_dims));
    return stage;
}

}  // namespace vpu
