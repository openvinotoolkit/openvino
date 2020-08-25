// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/utils/numeric.hpp>

#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <vector>
#include <string>
#include <memory>
#include <utility>

namespace vpu {

namespace {

using InputEdges = details::ContainerRange<StageInputVector, false>;

DimsOrder getMostSuitableOrder(const InputEdges& inputEdges) {
    DimsOrderMap<int> dimsOrderVotes;
    for (const auto& inEdge : inputEdges) {
        dimsOrderVotes[inEdge->input()->desc().dimsOrder()]++;
    }

    // Select DimsOrder with most votes.
    // For equal votes : HCW > CHW > HWC.

    DimsOrder finalOrder;
    int curVotes = -1;
    for (const auto& p : dimsOrderVotes) {
        if (p.second > curVotes) {
            finalOrder = p.first;
            curVotes = p.second;
        } else if (p.second == curVotes) {
            if (p.first.numDims() >= 3) {
                if (p.first.dimInd(Dim::C) == 2) {
                    finalOrder = p.first;
                } else if (p.first.dimInd(Dim::C) == 3 &&
                           finalOrder.dimInd(Dim::C) != 2) {
                    finalOrder = p.first;
                }
            }
        }
    }

    VPU_INTERNAL_CHECK(finalOrder.numDims() > 0,
                       "getMostSuitableOrder must find order with rank which is grater than 0, "
                       "actually rank is {}", finalOrder.numDims());
    VPU_INTERNAL_CHECK(curVotes > 0,
                       "getMostSuitableOrder: final order must have at least 1 vote "
                       "actually votes number is {}", curVotes);

    return finalOrder;
}

//
// StubConcatStage will be replaced with Data <-> Data edges on special stage processor
//

class StubConcatStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<StubConcatStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto finalOrder = getMostSuitableOrder(inputEdges());

        for (const auto& inEdge : inputEdges()) {
            orderInfo.setInput(inEdge, finalOrder);
        }

        orderInfo.setOutput(outputEdge(0), finalOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto output = outputEdge(0)->output();

        auto dimsOrder = output->desc().dimsOrder();

        //
        // Get smallest Dim over which Concat is done.
        //

        auto minConcatDimInd = dimsOrder.numDims() - 1;

        for (const auto& inEdge : inputEdges()) {
            auto input = inEdge->input();

            for (const auto& p : output->desc().dims()) {
                if (input->desc().dim(p.first) != p.second) {
                    minConcatDimInd = std::min(minConcatDimInd, dimsOrder.dimInd(p.first));
                }
            }
        }

        VPU_INTERNAL_CHECK(minConcatDimInd < dimsOrder.numDims(),
                           "{} stage with name {} must have minConcatDimInd no greater than number "
                           "of dimensions, actually index is {}, number of dimension is {}",
                           type(), name(), minConcatDimInd, dimsOrder.numDims());

        //
        // Initial StridesRequirement for inputs and output.
        //

        auto outputReqs = output->requiredStrides();

        auto inputReqs = outputReqs;
        for (int i = minConcatDimInd + 1; i < dimsOrder.numDims(); ++i) {
            inputReqs.remove(i);
        }

        //
        // Merge input StridesRequirement.
        //

        for (const auto& inEdge : inputEdges()) {
            auto curInput = inEdge->input();
            auto curInputReqs = curInput->requiredStrides();

            for (int i = 0; i < minConcatDimInd + 1; ++i) {
                if (outputReqs.get(i) == DimStride::Any) {
                    if (curInputReqs.get(i) != DimStride::Any) {
                        inputReqs.add(i, curInputReqs.get(i));
                        outputReqs.add(i, curInputReqs.get(i));
                    }
                }
            }
        }

        //
        // Merge output consumers StridesRequirement.
        //

        for (const auto& consumerEdge : output->consumerEdges()) {
            const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

            if (consumerInfo.hasInput(consumerEdge)) {
                const auto& consumerReqs = consumerInfo.getInput(consumerEdge);

                for (int i = 0; i < minConcatDimInd + 1; ++i) {
                    if (outputReqs.get(i) == DimStride::Any) {
                        if (consumerReqs.get(i) != DimStride::Any) {
                            inputReqs.add(i, consumerReqs.get(i));
                            outputReqs.add(i, consumerReqs.get(i));
                        }
                    }
                }
            }
        }

        //
        // Return merged StridesRequirement.
        //

        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, inputReqs);
        }
        stridesInfo.setOutput(outputEdge(0), outputReqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        VPU_INTERNAL_CHECK(numInputs() > 0,
                           "{} stage with name {} must have no less than 1 input, "
                           "actually provided {} inputs", type(), name(), numInputs());
        VPU_INTERNAL_CHECK(numOutputs() == 1,
                           "{} stage with name {} must have only 1 output, "
                           "actually provided {} outputs", type(), name(), numOutputs());

        const auto& firstInputPrecision = input(0)->desc().type();
        assertAllInputsOutputsTypes(this, {firstInputPrecision}, {firstInputPrecision});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_FORMAT("{} stage with name {} must never call serializeParamsImpl",
                         type(), name());
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_FORMAT("{} stage with name {} must never call serializeDataImpl",
                         type(), name());
    }
};

//
// ConcatStage will be inferred on device side
//

class ConcatStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConcatStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto finalOrder = getMostSuitableOrder(inputEdges());

        for (const auto& inEdge : inputEdges()) {
            orderInfo.setInput(inEdge, finalOrder);
        }

        orderInfo.setOutput(outputEdge(0), finalOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement().remove(0));
        }
        stridesInfo.setOutput(outputEdge(0), StridesRequirement().remove(0));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        VPU_INTERNAL_CHECK(numInputs() > 0,
                           "{} stage with name {} must have no less than 1 input, "
                           "actually provided {} inputs", type(), name(), numInputs());
        VPU_INTERNAL_CHECK(numOutputs() == 1,
                           "{} stage with name {} must have only 1 output, "
                           "actually provided {} outputs", type(), name(), numOutputs());

        const auto& firstInputPrecision = input(0)->desc().type();
        assertAllInputsOutputsTypes(this, {firstInputPrecision}, {firstInputPrecision});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto axis = attrs().get<Dim>("axis");
        const auto axisInd = input(0)->desc().dimsOrder().dimInd(axis);

        serializer.append(static_cast<uint32_t>(axisInd));
        serializer.append(static_cast<uint32_t>(numInputs()));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        for (const auto& input : inputs()) {
            input->serializeBuffer(serializer);
        }
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseConcat(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) const {
    VPU_THROW_UNLESS(!inputs.empty(),
                     "{} layer with name {} must have no less than 1 input, "
                     "actually provided 0 inputs", layer->type, layer->name);
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "{} layer with name {} must have only 1 output, actually provided {} outputs",
                     layer->type, layer->name, outputs.size());

    auto output = outputs[0];

    VPU_THROW_UNLESS(layer != nullptr,
                     "parseConcat expects valid CNNLayerPtr, got nullptr");
    auto concat = std::dynamic_pointer_cast<ie::ConcatLayer>(layer);
    VPU_THROW_UNLESS(concat != nullptr,
                     "{} layer with name {} must be able to convert to ie::ConcatLayer",
                     layer->type, layer->name);

    VPU_THROW_UNLESS(concat->_axis < output->desc().numDims(),
                     "{} layer with name {} must have axis attribute no grater than number of "
                     "dimensions, actually provided axis = {}, numDims = {}",
                     layer->type, layer->name, concat->_axis, output->desc().numDims());

    auto perm = DimsOrder::fromNumDims(output->desc().numDims()).toPermutation();
    auto axis = perm[output->desc().numDims() - 1 - concat->_axis];

    // If there is DSR as concat's output in the transformed graph, then we need to infer
    // concat on the device side. In other cases StubConcat stage will be added and it will
    // be replace with Data <-> Data edges.
    auto inferRequirement = ConcatInferRequirement::CanBeReplaced;
    if (auto concatOp = std::dynamic_pointer_cast<ngraph::opset3::Concat>(layer->getNode())) {
        inferRequirement = concatOp->get_input_source_output(0).get_node_shared_ptr()->get_type_info() ==
                           ngraph::vpu::op::DynamicShapeResolver::type_info
                           ? ConcatInferRequirement::NeedToInfer
                           : ConcatInferRequirement::CanBeReplaced;
    }

    _stageBuilder->addConcatStage(model, concat->name, concat, axis, inputs, output, inferRequirement);
}

Stage StageBuilder::addConcatStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        Dim axis,
        const DataVector& inputs,
        const Data& output,
        ConcatInferRequirement inferRequirement) {
    std::vector<DimValues> offsets;
    offsets.reserve(inputs.size());

    Stage stage;
    if (inferRequirement == ConcatInferRequirement::NeedToInfer) {
        stage = model->addNewStage<ConcatStage>(
                layer->name,
                StageType::Concat,
                layer,
                inputs,
                {output});
    } else {
        DimValues curOffset({{axis, 0}});
        for (const auto& input : inputs) {
            offsets.emplace_back(curOffset);
            curOffset.set(axis, curOffset[axis] + input->desc().dim(axis));
        }

        stage = addConcatStage(model, name, layer, std::move(offsets), inputs, output);
    }

    stage->attrs().set("axis", axis);

    return stage;
}

Stage StageBuilder::addConcatStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        std::vector<DimValues>&& offsets,
        const DataVector& inputs,
        const Data& output) {
    VPU_INTERNAL_CHECK(offsets.size() == inputs.size(),
                       "offsets count (provided {}) must be equal to inputs count (provided {}) to "
                       "create Concat stage with name {}", offsets.size(), inputs.size(), name);

    auto stage = model->addNewStage<StubConcatStage>(
        name,
        StageType::StubConcat,
        layer,
        inputs,
        {output});

    stage->attrs().set("offsets", std::move(offsets));

    return stage;
}

}  // namespace vpu
