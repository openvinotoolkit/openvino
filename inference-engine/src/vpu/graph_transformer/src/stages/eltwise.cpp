// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>
#include <limits>
#include <algorithm>

#include <vpu/utils/numeric.hpp>

namespace vpu {

namespace {

class EltwiseStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<EltwiseStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        if (_type != StageType::Prod &&
            step == ScalePropagationStep::Propagate) {
            // Keep the largest input scale factor.
            auto maxScale = std::numeric_limits<float>::lowest();
            for (const auto& inEdge : _inputEdges) {
                maxScale = std::max(maxScale, inputScales.at(inEdge->input()));
            }

            for (const auto& inEdge : _inputEdges) {
                auto curScale = inputScales.at(inEdge->input());

                if (!isFloatEqual(curScale, maxScale)) {
                    out[inEdge->input()] = maxScale / curScale;
                }
            }

            out[output] = maxScale;
        } else {
            // Eltwise can only propagate scaling for Sum and Max cases.
            for (const auto& inEdge : _inputEdges) {
                out[inEdge->input()] = 1.0f;
            }

            out[output] = 1.0f;
        }

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        auto in0Desc = input0->desc();
        auto in1Desc = input1->desc();
        auto outDesc = output->desc();

        auto finalOrder  = in0Desc.numDims() >= in1Desc.numDims() ? in0Desc.dimsOrder() : in1Desc.dimsOrder();
        auto secondOrder = in0Desc.numDims() >= in1Desc.numDims() ? in1Desc.dimsOrder() : in0Desc.dimsOrder();
        if (secondOrder.numDims() >= 3) {
            if (secondOrder.dimInd(Dim::C) == 1 /*HCW*/) {
                finalOrder = secondOrder;
            } else if (secondOrder.dimInd(Dim::C) == 2 /*CHW*/ && finalOrder.dimInd(Dim::C) != 1 /*HCW*/) {
                finalOrder = secondOrder;
            }
        }
        if (outDesc.numDims() > finalOrder.numDims())
            finalOrder = outDesc.dimsOrder();

        DataMap<DimsOrder> out;

        out[input0] = finalOrder.numDims() == in0Desc.numDims() ? finalOrder : in0Desc.dimsOrder();
        out[input1] = finalOrder.numDims() == in1Desc.numDims() ? finalOrder : in1Desc.dimsOrder();
        out[output] = finalOrder;

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        return DataMap<StridesRequirement>();
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto coeff1 = attrs().getOrDefault<float>("coeff1", 1.0f);
        auto coeff2 = attrs().getOrDefault<float>("coeff2", 1.0f);

        serializer.append(static_cast<float>(coeff1));
        serializer.append(static_cast<float>(coeff2));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeNewBuffer(serializer, output->desc().dimsOrder());
        output->serializeNewBuffer(serializer);
        input1->serializeNewBuffer(serializer, output->desc().dimsOrder());
    }
};

}  // namespace

void FrontEnd::parseEltwise(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() >= 2);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::EltwiseLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto stageType = StageType::None;
    auto subCoefficient = 1.0f;
    switch (layer->_operation) {
    case ie::EltwiseLayer::eOperation::Sum:
        stageType = StageType::Sum;
        break;
    case ie::EltwiseLayer::eOperation::Prod:
        stageType = StageType::Prod;
        break;
    case ie::EltwiseLayer::eOperation::Max:
        stageType = StageType::Max;
        break;
    case ie::EltwiseLayer::eOperation::Sub:
        if (inputs.size() > 2) {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " with multiple inputs is not supported";
        }
        stageType = StageType::Sum;
        subCoefficient = -1.f;
        break;
    default:
        VPU_THROW_EXCEPTION << "Eltwise operation" << layer->_operation << " is not supported";
    }

    if (stageType != StageType::Sum && !layer->coeff.empty()) {
        VPU_THROW_EXCEPTION << layer->name << " coefficients only supported for Sum/Sub operations.";
    }

    auto output = outputs[0];

    auto tempOutput = output;
    if (inputs.size() > 2) {
        tempOutput = model->duplicateData(
            output,
            formatString("@temp@1/%d", inputs.size() - 2));
    }

    DataVector tempInputs(2);
    tempInputs[0] = inputs[0];
    tempInputs[1] = inputs[1];

    auto stage = model->addNewStage<EltwiseStage>(
        layer->name,
        stageType,
        layer,
        tempInputs,
        {tempOutput});

    if (layer->coeff.size() > 0) {
        stage->attrs().set<float>("coeff1", layer->coeff[0]);
    }
    if (layer->coeff.size() > 1 || subCoefficient != 1.0f) {
        stage->attrs().set<float>("coeff2", subCoefficient * (layer->coeff.size() > 1 ? layer->coeff[1] : 1.0f));
    }

    tempInputs[0] = tempOutput;
    for (int ind = 2; ind < inputs.size(); ++ind) {
        tempInputs[1] = inputs[ind];

        if (ind + 1 == inputs.size()) {
            tempOutput = output;
        } else {
            tempOutput = model->duplicateData(
                output,
                formatString("@temp@%d/%d", ind, inputs.size() - 2));
        }

        stage = model->addNewStage<EltwiseStage>(
            layer->name + "@" + std::to_string(ind - 1),
            stageType,
            layer,
            tempInputs,
            {tempOutput});

        if (layer->coeff.size() > ind) {
            stage->attrs().set<float>("coeff2", layer->coeff[ind]);
        }

        tempInputs[0] = tempOutput;
    }
}

Stage StageBuilder::addSumStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    return model->addNewStage<EltwiseStage>(
        name,
        StageType::Sum,
        layer,
        {input0, input1},
        {output});
}

}  // namespace vpu
