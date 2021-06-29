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

#define MAP_ELEMENTS(op, f) {vpu::EltwiseOperation::op, &f<StageType::op>}


// void FrontEnd::parseSubtract(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
//     auto subtract = ngraph::as_type_ptr<ngraph::opset4::Subtract>(node);
//     VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
//     auto eltwiseOp = EltwiseOperation::Sub;
//     parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
    
// }

namespace vpu {

namespace {
template<StageType T>
StageType onlyOneInput(EltwiseOperation op, size_t input_size) {
    if (input_size != 1) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only one input";
    }
    return T;
}

template<StageType T>
StageType onlyTwoInputs(EltwiseOperation op, size_t input_size) {
    if (input_size != 2) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only two inputs";
    }
    return T;
}

template<StageType T>
StageType moreThanOneInput(EltwiseOperation op, size_t input_size) {
    if (input_size < 2) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports two inputs and more";
    }
    return T;
}

template<StageType T>
StageType onlyThreeInputs(EltwiseOperation op, size_t input_size) {
    if (input_size != 3) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only three inputs";
    }
    return T;
}

static const std::map<EltwiseOperation, std::function<StageType(EltwiseOperation, size_t)>> eltwise_map = {
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
                    StageType::Logical_AND
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


// Rework it by using macroses
void FrontEnd::parseSubtract(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Subtract>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Sub;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
    
}

void FrontEnd::parseAdd(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Add>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Sum;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}

void FrontEnd::parseMultiply(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Multiply>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Prod;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseMaximum(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Maximum>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Max;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseDivide(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Divide>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Div;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseMinimum(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Minimum>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Min;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseSquaredDifference(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::SquaredDifference>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Squared_diff;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Equal>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Equal;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseNotEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::NotEqual>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Not_equal;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseGreater(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Greater>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Greater;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseGreaterEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::GreaterEqual>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Greater_equal;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseLess(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::Less>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Less;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseLessEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::LessEqual>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Less_equal;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseLogicalNot(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::LogicalNot>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Logical_NOT;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseLogicalAnd(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::LogicalAnd>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Logical_AND;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseLogicalOr(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::LogicalOr>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Logical_OR;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}
void FrontEnd::parseLogicalXor(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto subtract = ngraph::as_type_ptr<ngraph::opset4::LogicalXor>(node);
    VPU_THROW_UNLESS(subtract != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto eltwiseOp = EltwiseOperation::Logical_XOR;
    parseEltwiseImpl(model, node, inputs, outputs, eltwiseOp);
}

void FrontEnd::parseEltwiseImpl(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs, const EltwiseOperation eltwiseOperation) const {

    IE_ASSERT(outputs.size() == 1);
    auto stageType = StageType::None;
    auto subCoefficient = 1;

    if (eltwiseOperation == EltwiseOperation::Sub) {
        if (inputs.size() != 2) {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << node->get_type_name() << " with multiple inputs is not supported";
        }
        stageType = StageType::Sum;
        subCoefficient = -1;
    }
    //  else if (/*layer->_operation == ie::EltwiseLayer::eOperation::Mean*/1) {
    //     if (inputs.size() != 2) {
    //         VPU_THROW_EXCEPTION << "Eltwise operation: " << eltwiseOperation << " with multiple inputs is not supported";
    //     }
    //     stageType = StageType::Sum;
    else {
        if (eltwise_map.find(eltwiseOperation) != eltwise_map.end()) {
            stageType = eltwise_map.at(eltwiseOperation)(eltwiseOperation, inputs.size());
        } else {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << eltwiseOperation << " is not supported";
        }
    }

    // if (stageType != StageType::Sum && !layer->coeff.empty()) {
    //     VPU_THROW_EXCEPTION << layer->name << " coefficients only supported for Sum/Sub operations.";
    // }

    auto output = outputs[0];

    auto tempOutput = output;
    if (inputs.size() > 2) {
        tempOutput = model->duplicateData(
            output,
            formatString("@temp@1/%d", inputs.size() - 2));
    }

    DataVector tempInputs(3);
    tempInputs[0] = inputs[0];

    if (stageType == StageType::Logical_NOT)
        tempInputs[1] = model->addFakeData();
    else
        tempInputs[1] = inputs[1];

    tempInputs[2] = model->addFakeData();

    auto stage = model->addNewStage<EltwiseStage>(node->get_friendly_name(), stageType, node, tempInputs, {tempOutput});

    const auto& type = inputs.front()->desc().type();
    IE_ASSERT(type == DataType::FP16 || type == DataType::S32);

    // if (eltwiseOperation == ie::EltwiseLayer::eOperation::Mean) {
    //     // Mean supports only FP16
    //     auto mean = ngraph::as_type_ptr<ngraph::opset4::Add>(node);
    //     IE_ASSERT(type == DataType::FP16);
    //     stage->attrs().set<float>("coeff1",  0.5);
    //     stage->attrs().set<float>("coeff2",  0.5);
    // } else {
    //     if (layer->coeff.size() > 0) {
    //         if (type == DataType::FP16) {
    //             stage->attrs().set<float>("coeff1", layer->coeff[0]);
    //         } else {
    //             stage->attrs().set<std::int32_t>("coeff1", static_cast<int32_t>(layer->coeff[0]));
    //         }
    //     }
    //     if (layer->coeff.size() > 1 || subCoefficient != 1) {
    //         if (type == DataType::FP16) {
    //             stage->attrs().set<float>("coeff2", subCoefficient * (layer->coeff.size() > 1 ? layer->coeff[1] : 1.0f));
    //         } else {
    //             stage->attrs().set<std::int32_t>("coeff2", subCoefficient * (layer->coeff.size() > 1 ? static_cast<int32_t>(layer->coeff[1]) : 1));
    //         }
    //     }
    // }

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
            node->get_friendly_name() + "@" + std::to_string(ind - 1),
            stageType,
            node,
            tempInputs,
            {tempOutput});

        // if (layer->coeff.size() > ind) {
        //     stage->attrs().set<float>("coeff2", layer->coeff[ind]);
        // }

        tempInputs[0] = tempOutput;
    }
}

void FrontEnd::parseSelect(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto select = ngraph::as_type_ptr<ngraph::opset4::Select>(node);
    VPU_THROW_UNLESS(select != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    if (inputs.size() != 3) {
        VPU_THROW_EXCEPTION << "Select supports only three inputs";
    }

    auto stage = model->addNewStage<EltwiseStage>(select->get_friendly_name(), StageType::Select, select, inputs, outputs);
}

Stage StageBuilder::addSumStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    const Data& fakeInput2 = model->addFakeData();
    return model->addNewStage<EltwiseStage>(
        name,
        StageType::Sum,
        node,
        {input0, input1, fakeInput2},
        {output});
}

Stage StageBuilder::addProdStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    const Data& fakeInput2 = model->addFakeData();
    return model->addNewStage<EltwiseStage>(
            name,
            StageType::Prod,
            node,
            {input0, input1, fakeInput2},
            {output});
}

Stage StageBuilder::addMaxStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    const Data& fakeInput2 = model->addFakeData();
    return model->addNewStage<EltwiseStage>(
        name,
        StageType::Max,
        node,
        {input0, input1, fakeInput2},
        {output});
}

}  // namespace vpu
