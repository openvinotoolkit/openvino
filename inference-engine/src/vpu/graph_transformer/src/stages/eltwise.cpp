// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>
#include <map>
#include <limits>
#include <algorithm>

#include <vpu/utils/numeric.hpp>

#define MAP_ELEMENTS(op, f) {InferenceEngine::EltwiseLayer::eOperation::op, &f<StageType::op>}

namespace vpu {

namespace {

template<StageType T>
StageType onlyOneInput(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size != 1) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only one input";
    }
    return T;
}

template<StageType T>
StageType onlyTwoInputs(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size != 2) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only two inputs";
    }
    return T;
}

template<StageType T>
StageType moreThanOneInput(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size < 2) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports two inputs and more";
    }
    return T;
}

template<StageType T>
StageType onlyThreeInputs(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size != 3) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only three inputs";
    }
    return T;
}

static const std::map<ie::EltwiseLayer::eOperation, std::function<StageType(ie::EltwiseLayer::eOperation, size_t)>> eltwise_map = {
        MAP_ELEMENTS(Sum,           moreThanOneInput),
        MAP_ELEMENTS(Prod,          moreThanOneInput),
        MAP_ELEMENTS(Max,           moreThanOneInput),
        MAP_ELEMENTS(Div,           onlyTwoInputs),
        MAP_ELEMENTS(Min,           moreThanOneInput),
        MAP_ELEMENTS(Squared_diff,  onlyTwoInputs),
        MAP_ELEMENTS(Equal,         onlyTwoInputs),
        MAP_ELEMENTS(Not_equal,     onlyTwoInputs),
        MAP_ELEMENTS(Greater,       onlyTwoInputs),
        MAP_ELEMENTS(Greater_equal, onlyTwoInputs),
        MAP_ELEMENTS(Less,          onlyTwoInputs),
        MAP_ELEMENTS(Less_equal,    onlyTwoInputs),
        MAP_ELEMENTS(Logical_NOT,   onlyOneInput),
        MAP_ELEMENTS(Logical_AND,   moreThanOneInput),
        MAP_ELEMENTS(Logical_OR,    moreThanOneInput),
        MAP_ELEMENTS(Logical_XOR,   moreThanOneInput),
        MAP_ELEMENTS(Pow,           onlyTwoInputs),
        MAP_ELEMENTS(Floor_mod,     onlyTwoInputs),
        MAP_ELEMENTS(Abs,           onlyOneInput),
};

class EltwiseStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<EltwiseStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto input2 = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        auto in0Desc = input0->desc();
        auto in1Desc = input1->desc();
        auto in2Desc = input2->desc();
        auto outDesc = output->desc();

        auto finalOrder  = in0Desc.numDims() >= in1Desc.numDims() ? in0Desc.dimsOrder() : in1Desc.dimsOrder();
        auto secondOrder = in0Desc.numDims() >= in1Desc.numDims() ? in1Desc.dimsOrder() : in0Desc.dimsOrder();
        auto thirdOrder = in0Desc.numDims() >= in2Desc.numDims() ? in2Desc.dimsOrder() : in0Desc.dimsOrder();
        if (secondOrder.numDims() >= 3) {
            if (secondOrder.dimInd(Dim::C) == 1 /*HCW*/) {
                finalOrder = secondOrder;
            } else if (secondOrder.dimInd(Dim::C) == 2 /*CHW*/ && finalOrder.dimInd(Dim::C) != 1 /*HCW*/) {
                finalOrder = secondOrder;
            }
        }
        if (thirdOrder.numDims() >= 3) {
            if (thirdOrder.dimInd(Dim::C) == 1 /*HCW*/) {
                finalOrder = thirdOrder;
            } else if (thirdOrder.dimInd(Dim::C) == 2 /*CHW*/ && finalOrder.dimInd(Dim::C) != 1 /*HCW*/) {
                finalOrder = thirdOrder;
            }
        }
        if (outDesc.numDims() > finalOrder.numDims()) {
            finalOrder = outDesc.dimsOrder();
        }

        orderInfo.setInput(inputEdge(0), finalOrder.numDims() == in0Desc.numDims() ? finalOrder : in0Desc.dimsOrder());
        orderInfo.setInput(inputEdge(1), finalOrder.numDims() == in1Desc.numDims() ? finalOrder : in1Desc.dimsOrder());
        orderInfo.setInput(inputEdge(2), finalOrder.numDims() == in2Desc.numDims() ? finalOrder : in2Desc.dimsOrder());
        orderInfo.setOutput(outputEdge(0), finalOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        const auto& operation = type();
        const auto& dataTypeInput0 = input(0)->desc().type();
        const auto& dataTypeOutput = output(0)->desc().type();

        {
            static const std::set<StageType> stageTypesWhichSupportS32 = {
                    StageType::Sum,
                    StageType::Greater_equal,
                    StageType::Equal,
                    StageType::Select,
                    StageType::Prod,
                    StageType::Max,
                    StageType::Div,
                    StageType::Min,
                    StageType::Logical_NOT,
                    StageType::Logical_AND,
                    StageType::Abs,
            };
            auto supportedDataTypesInput0 = EnumSet<DataType>{DataType::FP16};
            if (stageTypesWhichSupportS32.count(operation)) {
                supportedDataTypesInput0.insert(DataType::S32);
            }

            VPU_THROW_UNLESS(supportedDataTypesInput0.count(dataTypeInput0) != 0,
                "Stage node %v types check error: input #0 has type %v, but one of %v is expected",
                static_cast<Handle<StageNode>>(this), dataTypeInput0, supportedDataTypesInput0);
        }

        if (operation == StageType::Select && dataTypeInput0 == DataType::S32) {
            auto supportedDataTypesInput1 = EnumSet<DataType>{DataType::FP16, DataType::S32};
            const auto& dataTypeInput1 = input(1)->desc().type();
            VPU_THROW_UNLESS(supportedDataTypesInput1.count(dataTypeInput1) != 0,
                             "Stage node %v types check error: input #1 has type %v, but one of %v is expected",
                             static_cast<Handle<StageNode>>(this), dataTypeInput1, supportedDataTypesInput1);

            assertInputsOutputsTypes(this, {{dataTypeInput0}, {dataTypeInput1}, {dataTypeInput1}}, {{dataTypeInput1}});
        } else if ((operation == StageType::Greater || operation == StageType::Less || operation == StageType::Equal)
                        && dataTypeInput0 != dataTypeOutput) {
            assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}}, {{DataType::S32}});
        } else {
            assertInputsOutputsTypes(this, {{dataTypeInput0}, {dataTypeInput0}, {dataTypeInput0}}, {{dataTypeInput0}});
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& type = input(0)->desc().type();

        if (type == DataType::FP16) {
            serializer.append(attrs().getOrDefault<float>("coeff1", 1.0f));
            serializer.append(attrs().getOrDefault<float>("coeff2", 1.0f));
        } else if (type == DataType::S32) {
            serializer.append(attrs().getOrDefault<std::int32_t>("coeff1", 1));
            serializer.append(attrs().getOrDefault<std::int32_t>("coeff2", 1));
        } else {
             IE_THROW() << type << " isn't supported";
        }

        auto postOperation = attrs().getOrDefault<StageType>("postOperation", StageType::Empty);
        serializer.append(static_cast<int>(postOperation));

        if (type == DataType::FP16) {
            serializer.append(attrs().getOrDefault<float>("negativeSlope", 0.0f));
            serializer.append(attrs().getOrDefault<float>("min_value", 0.0f));
            serializer.append(attrs().getOrDefault<float>("max_value", 1.0f));
        } else {
            serializer.append(attrs().getOrDefault<std::int32_t>("negativeSlope", 0));
            serializer.append(attrs().getOrDefault<std::int32_t>("min_value", 0));
            serializer.append(attrs().getOrDefault<std::int32_t>("max_value", 1));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto input2 = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseEltwise(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    auto layer = std::dynamic_pointer_cast<ie::EltwiseLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(outputs.size() == 1);

    auto stageType = StageType::None;
    auto subCoefficient = 1;

    if (layer->_operation == ie::EltwiseLayer::eOperation::Sub) {
        if (inputs.size() != 2) {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " with multiple inputs is not supported";
        }
        stageType = StageType::Sum;
        subCoefficient = -1;
    } else if (layer->_operation == ie::EltwiseLayer::eOperation::Mean) {
        if (inputs.size() != 2) {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " with multiple inputs is not supported";
        }
        stageType = StageType::Sum;
    } else {
        if (eltwise_map.find(layer->_operation) != eltwise_map.end()) {
            stageType = eltwise_map.at(layer->_operation)(layer->_operation, inputs.size());
        } else {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " is not supported";
        }
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

    DataVector tempInputs(3);
    tempInputs[0] = inputs[0];

    if (stageType == StageType::Logical_NOT || stageType == StageType::Abs)
        tempInputs[1] = model->addFakeData();
    else
        tempInputs[1] = inputs[1];

    tempInputs[2] = model->addFakeData();

    auto stage = model->addNewStage<EltwiseStage>(layer->name, stageType, layer, tempInputs, {tempOutput});

    const auto& type = inputs.front()->desc().type();
    IE_ASSERT(type == DataType::FP16 || type == DataType::S32);

    if (layer->_operation == ie::EltwiseLayer::eOperation::Mean) {
        // Mean supports only FP16
        IE_ASSERT(type == DataType::FP16);
        stage->attrs().set<float>("coeff1",  0.5);
        stage->attrs().set<float>("coeff2",  0.5);
    } else {
        if (layer->coeff.size() > 0) {
            if (type == DataType::FP16) {
                stage->attrs().set<float>("coeff1", layer->coeff[0]);
            } else {
                stage->attrs().set<std::int32_t>("coeff1", static_cast<int32_t>(layer->coeff[0]));
            }
        }
        if (layer->coeff.size() > 1 || subCoefficient != 1) {
            if (type == DataType::FP16) {
                stage->attrs().set<float>("coeff2", subCoefficient * (layer->coeff.size() > 1 ? layer->coeff[1] : 1.0f));
            } else {
                stage->attrs().set<std::int32_t>("coeff2", subCoefficient * (layer->coeff.size() > 1 ? static_cast<int32_t>(layer->coeff[1]) : 1));
            }
        }
    }

    stage->attrs().set<StageType>("postOperation", StageType::Empty);

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

void FrontEnd::parseSelect(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    auto layer = std::dynamic_pointer_cast<ie::SelectLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    if (inputs.size() != 3) {
        VPU_THROW_EXCEPTION << "Select supports only three inputs";
    }

    auto stage = model->addNewStage<EltwiseStage>(layer->name, StageType::Select, layer, inputs, outputs);
}

Stage StageBuilder::addSumStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    const Data& fakeInput2 = model->addFakeData();
    return model->addNewStage<EltwiseStage>(
        name,
        StageType::Sum,
        layer,
        {input0, input1, fakeInput2},
        {output});
}

Stage StageBuilder::addProdStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    const Data& fakeInput2 = model->addFakeData();
    return model->addNewStage<EltwiseStage>(
            name,
            StageType::Prod,
            layer,
            {input0, input1, fakeInput2},
            {output});
}

Stage StageBuilder::addMaxStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    const Data& fakeInput2 = model->addFakeData();
    return model->addNewStage<EltwiseStage>(
        name,
        StageType::Max,
        layer,
        {input0, input1, fakeInput2},
        {output});
}

}  // namespace vpu
