// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <low_precision/network_helper.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <numeric>

#include <ngraph/rt_info.hpp>
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

// Return true if `type` can be castable to at least one of `type`
bool NetworkHelper::is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types) {
    for (auto another : types) {
        if (type.is_castable(another)) {
            return true;
        }
    }
    return false;
}

bool NetworkHelper::notAllChildrensAreFQ(const NodeVector& childrens) {
    // NOTE: This check was added for models that don't have FQ after AvgPool
    //       They will have transparent precision as it was in old LPT.
    for (const auto& child : childrens) {
        if (!is_type<opset1::FakeQuantize>(child)) {
            return true;
        }
    }
    return false;
}

// Collect and return a vector with all nodes that consumes any of the `node` output
std::vector<Input<Node>> NetworkHelper::consumer_inputs(std::shared_ptr<Node> node) {
    std::vector<Input<Node>> result;
    for (const auto& output_port : node->outputs()) {
        for (const auto &input : output_port.get_target_inputs()) {
            result.push_back(input);
        }
    }
    return result;
}

std::vector<std::shared_ptr<Node>> NetworkHelper::consumers(std::shared_ptr<Node> node) {
    auto inputs = consumer_inputs(node);
    std::vector<std::shared_ptr<Node>> result(inputs.size());
    std::transform(inputs.begin(), inputs.end(), result.begin(), [](Input<Node> input){ return input.get_node()->shared_from_this(); });
    return result;
}

bool NetworkHelper::isConstantPath(const std::shared_ptr<Node>& op) {
    const auto isNotConstantPathOperation = [](const std::shared_ptr<Node>& node) -> bool {
        return is_type<opset1::Parameter>(node) ||
            is_type<opset1::Convolution>(node) ||
            is_type<opset1::GroupConvolution>(node) ||
            is_type<opset1::MatMul>(node) ||
            is_type<opset1::ConvolutionBackpropData>(node);
    };

    if (isNotConstantPathOperation(op)) {
        return false;
    }

    std::queue<Input<Node>> inputs;
    const std::vector<Input<Node>> nodeInputs = op->inputs();
    for (const Input<Node>& nodeInput : nodeInputs) {
        inputs.push(nodeInput);
    }

    while (!inputs.empty()) {
        Input<Node> input = inputs.front();
        inputs.pop();

        const Output<Node>& sourceOutput = input.get_source_output();
        const auto parentNode = sourceOutput.get_node_shared_ptr();
        if (isNotConstantPathOperation(parentNode)) {
            return false;
        }

        for (size_t inputIndex = 0; inputIndex < parentNode->get_input_size(); ++inputIndex) {
            inputs.push(parentNode->input(inputIndex));
        }
    }
    return true;
}

std::shared_ptr<opset1::Constant> NetworkHelper::foldDequantizationConstant(
    const std::shared_ptr<opset1::Constant>& foldingConstant,
    const std::shared_ptr<Node>& operation,
    const size_t outIdx) {
    OutputVector inputs = operation->input_values();
    OutputVector outputs(operation->get_output_size());

    if (shape_size(foldingConstant->get_shape()) == 1ul) {
        return toScalar(foldingConstant);
    } else {
        inputs[0] = foldingConstant;
        const auto op = operation->clone_with_new_inputs(inputs);

        if (std::dynamic_pointer_cast<op::TypeRelaxedBase>(op)) {
            setOutDataPrecisionForTypeRelaxed(op, inputs[0].get_element_type());
        }

        // constant folding of constant
        op->constant_fold(outputs, inputs);

        const auto result = as_type_ptr<opset1::Constant>(outputs[outIdx].get_node_shared_ptr());
        if (result == nullptr) {
            THROW_IE_LPT_EXCEPTION(*result) << "result of constant folding is not constant";
        }

        return result;
    }
}

size_t NetworkHelper::getOutputChannelsCount(std::shared_ptr<const Node> layer, bool isOnWeights) {
    if (layer->outputs().size() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer->get_friendly_name() << " doesn't have output tensors";
    }

    if (layer->outputs().size() > 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer->get_friendly_name() << " has too many output tensors, expected one";
    }

    PartialShape shape = layer->get_output_partial_shape(0);
    if (shape.rank() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid dimensions count (0) in output of " << layer->get_friendly_name() << " layer on weights";
    }
    if (isOnWeights) {
        return shape[0].get_length();
    } else {
        if (shape.rank() == 1) {
            return shape[0].get_length();
        }
        return shape[1].get_length();
    }
}

std::vector<std::shared_ptr<Node>> NetworkHelper::getParentsRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes,
        const int portIndex) {
    std::vector<std::shared_ptr<Node>> parents;
    int i = 0;
    for (auto input : layer->inputs()) {
        if ((portIndex == -1) || (portIndex == i)) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (is_castable_to_one_of(parent->get_type_info(), exceptionLayerTypes)) {
                const std::vector<std::shared_ptr<Node>> tmpParents = getParentsRecursivelyExceptTypes(parent, exceptionLayerTypes);
                parents.insert(parents.end(), tmpParents.begin(), tmpParents.end());
            } else {
                parents.push_back(parent);
            }
        }

        i++;
    }
    return parents;
}

size_t NetworkHelper::getInputChannelsCount(std::shared_ptr<Node> layer) {
    if (layer->get_input_size() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "There are no input layers";
    }

    PartialShape shape = layer->get_input_partial_shape(0);
    if (shape.rank().get_length() <= 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid dimensions count (0) in input of " << layer->get_friendly_name();
    }

    return shape[1].get_length();
}

size_t NetworkHelper::getGroupsCount(std::shared_ptr<Node> layer) {
    if (as_type_ptr<opset1::Convolution>(layer)) {
        return 1;
    } else if (auto group_convolution = as_type_ptr<opset1::GroupConvolution>(layer)) {
        return layer->get_input_shape(1)[0];    // input weights for opset1::GC is in format GOI..., see the specification
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid layer type of " << layer->get_friendly_name() << "; expected Convolutino or GroupConvolution";
    }
}

// Assumin tensor in NC... layout, append necessary number of 1s to shape to align it to a give rank
Shape NetworkHelper::alignShapeForChannelDim(const Shape& shape, Rank rank) {
    assert(shape.size() == 1);
    assert(rank.is_static());
    Shape result = shape;
    result.resize(rank.get_length() - 1, 1);
    return result;
}

void NetworkHelper::removeLayer(std::shared_ptr<Node> layer) {
    ngraph::replace_output_update_name(layer->output(0), layer->input_value(0));
}

std::shared_ptr<Node> NetworkHelper::swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply, const int multiplyBranch) {
    // Multiply --> Add(addAfterMultiply)  ==>  Add(new) --> Multiply(new)
    // That means x*a + b ==> (x + b/a)*a; tries to fold b/a
    const auto multiply = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch);

    const auto multiplyParent1 = multiply->get_input_node_shared_ptr(0);
    const auto multiplyParent2 = multiply->get_input_node_shared_ptr(1);

    auto multiplyInput = as_type_ptr<opset1::Multiply>(multiplyParent1);
    auto multiplyConst = as_type_ptr<opset1::Constant>(multiplyParent2);
    int multiplyInputBranch = 0;

    if (multiplyConst == nullptr) {
        multiplyInput = as_type_ptr<opset1::Multiply>(multiplyParent2);
        multiplyConst = as_type_ptr<opset1::Constant>(multiplyParent1);
        multiplyInputBranch = 1;
    }

    if (multiplyConst == nullptr)
        return addAfterMultiply;

    const auto x = multiply->get_input_source_output(multiplyInputBranch);
    auto a = multiply->get_input_node_shared_ptr(multiplyInputBranch == 0 ? 1 : 0);
    auto b = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch == 0 ? 1 : 0);
    std::shared_ptr<Node> bDivA;

    if (shape_size(b->get_output_shape(0)) == 1 ||
        shape_size(a->get_output_shape(0)) == 1 ||
        shape_size(b->get_output_shape(0)) == shape_size(a->get_output_shape(0))) {
        // safely division to avoid NaN
        const std::vector<float> bValues = as_type_ptr<opset1::Constant>(b)->cast_vector<float>();
        const std::vector<float> aValues = as_type_ptr<opset1::Constant>(a)->cast_vector<float>();
        const bool aBroadcasted = bValues.size() > aValues.size();
        const bool bBroadcasted = bValues.size() < aValues.size();
        std::vector<float> bDivAValues(aBroadcasted ? bValues.size() : aValues.size());

        for (size_t i = 0; i < bDivAValues.size(); ++i) {
            const auto bi = bValues[bBroadcasted ? 0 : i];
            const auto ai = aValues[aBroadcasted ? 0 : i];
            if (bi != 0.f || ai != 0.f) {
                bDivAValues[i] = bi / ai;
            } else {
                bDivAValues[i] = 0.f;
            }
        }

        // TODO: issue #49868
        auto aPrecision = a->get_output_element_type(0);
        bDivA = std::make_shared<opset1::Constant>(
                aPrecision,
                aBroadcasted ? b->get_output_shape(0) : a->get_output_shape(0),
                bDivAValues);
    } else {
        b = foldConvert(b, element::f32);
        a = foldConvert(a, element::f32);
        bDivA = fold<opset1::Divide>(b, a);
        // TODO: issue #49868
        bDivA = foldConvert(bDivA, a->get_output_element_type(0));
    }

    OutputVector inputs{ {}, {} };
    inputs[0] = x;
    inputs[1] = bDivA;

    std::shared_ptr<opset1::Add> newAdd = std::make_shared<op::TypeRelaxed<opset1::Add>>(
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{ x.get_element_type() },
        ngraph::op::TemporaryReplaceOutputType(inputs[0], element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(inputs[1], element::f32).get());
    copyInfo(addAfterMultiply, newAdd);

    auto newMultiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{ multiply->get_output_element_type(0) },
            ngraph::op::TemporaryReplaceOutputType(newAdd, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(a, element::f32).get());
    copyInfo(multiply, newMultiply);

    replace_node(addAfterMultiply, newMultiply);
    return newMultiply;
}

void NetworkHelper::copyInfo(const std::shared_ptr<Node>& source, const std::shared_ptr<Node>& target) {
    // TODO: merge_runtime_info with correctly defined DEQUANTIZATION
    const auto& sourceAttributes = source->get_rt_info();
    auto& targetAttrubutes = target->get_rt_info();
    for (auto attribute : sourceAttributes) {
        targetAttrubutes[attribute.first] = attribute.second;
    }

    const std::string friendlyName = source->get_friendly_name();
    if (!friendlyName.empty()) {
        target->set_friendly_name(friendlyName);
    }
}

void NetworkHelper::cleanRunTimeInfo(const std::shared_ptr<Node>& layer) {
    auto& rt_info = layer->get_rt_info();
    auto attributeIter = rt_info.find("DEQUANTIZATION");
    if (rt_info.find("DEQUANTIZATION") != rt_info.end()) {
        rt_info.erase(attributeIter);
    }
}

bool NetworkHelper::isScalarLike(std::shared_ptr<opset1::Constant> constant) {
    return constant->get_all_data_elements_bitwise_identical();
}

bool NetworkHelper::isZero(std::shared_ptr<opset1::Constant> constant) {
    static const float minQuantizationShift = 1e-32f;

    auto values = constant->cast_vector<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        if (fabs(values[i]) > minQuantizationShift) {
            return false;
        }
    }

    return true;
}

std::shared_ptr<opset1::Constant> NetworkHelper::toScalar(std::shared_ptr<opset1::Constant> constant) {
    assert(isScalarLike(constant));
    return std::make_shared<opset1::Constant>(constant->get_element_type(), Shape{}, constant->get_data_ptr());
}

std::shared_ptr<Node> NetworkHelper::getConstantInput(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> constant1 = as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr());
    if (!constant1) {
        constant1 = as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr());
    }
    return constant1;
}

int NetworkHelper::getConstantInputIndex(std::shared_ptr<Node> node) {
    if (as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(1)) != nullptr) {
        return 1;
    }

    if (as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(0)) != nullptr) {
        return 0;
    }

    return -1;
}

std::vector<size_t> NetworkHelper::updateReshapeValues(
    const Shape& elementwiseConstantShape,
    const Shape& elementwiseShape,
    const std::vector<size_t>& reshapeValues) {
    Shape updatedReshapeValues = reshapeValues;
    for (size_t elementwiseIndex = 0, reshapeIndex = 0; elementwiseIndex < elementwiseConstantShape.size(); ++elementwiseIndex) {
        if (elementwiseConstantShape[elementwiseIndex] != elementwiseShape[elementwiseIndex]) {
            size_t reducedValue = 1ul;
            for (; reshapeIndex < reshapeValues.size(); ++reshapeIndex) {
                reducedValue *= reshapeValues[reshapeIndex];
                updatedReshapeValues[reshapeIndex] = 1ul;
                if (reducedValue == elementwiseShape[elementwiseIndex]) {
                    reshapeIndex++;
                    break;
                }
            }
        } else {
            size_t reducedValue = 1ul;
            for (; reshapeIndex < reshapeValues.size(); ++reshapeIndex) {
                reducedValue *= reshapeValues[reshapeIndex];
                if (reducedValue == elementwiseConstantShape[elementwiseIndex]) {
                    reshapeIndex++;
                    break;
                }
            }
        }
    }
    return updatedReshapeValues;
}

std::shared_ptr<ngraph::opset1::Multiply> NetworkHelper::optimizeMultipliesAfter(std::shared_ptr<Node> node) {
    std::shared_ptr<ngraph::opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(std::move(node));
    if (!multiply) {
        THROW_IE_LPT_EXCEPTION(*multiply) << "Unexpected operation type";
    }

    if (multiply->output(0).get_target_inputs().size() == 1) {
        auto constant1 = getConstantInput(multiply);
        if (!constant1 || constant1->output(0).get_target_inputs().size() != 1) {
            return multiply;
        }

        auto nextMultiplyInput = *multiply->output(0).get_target_inputs().begin();
        auto nextMultiply = as_type_ptr<op::TypeRelaxed<opset1::Multiply>>(nextMultiplyInput.get_node()->shared_from_this());
        if (nextMultiply) {
            auto constant2 = getConstantInput(nextMultiply);
            if (!constant2 || constant2->output(0).get_target_inputs().size() != 1) {
                return multiply;
            }

            auto newInput = multiply->input_value(1 - constant1->output(0).get_target_inputs().begin()->get_index());
            auto newConst = fold<opset1::Multiply>(constant1, constant2);
            auto inputPrecision0 = nextMultiply->get_origin_input_type(0);
            auto inputPrecision1 = nextMultiply->get_origin_input_type(1);
            auto outputPrecision = nextMultiply->get_overridden_output_type(0);
            auto newMultiply =
                    std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
                            std::vector<element::Type>{ inputPrecision0, inputPrecision1 },
                            std::vector<element::Type>{ outputPrecision },
                            ngraph::op::TemporaryReplaceOutputType(newInput, inputPrecision0).get(),
                            ngraph::op::TemporaryReplaceOutputType(newConst, inputPrecision1).get());
            copy_runtime_info(multiply, newMultiply);
            replace_node(nextMultiply, newMultiply);
            return newMultiply;
        }
    }

    return nullptr;
}

std::shared_ptr<opset1::Constant> NetworkHelper::round(std::shared_ptr<Node> node, element::Type target_type) {
    const auto constant = as_type_ptr<opset1::Constant>(node);
    assert(constant);

    const auto castedConstant = as_type_ptr<ngraph::opset1::Constant>(fold<op::v0::Convert>(
        fold<ngraph::op::v5::Round>(constant->output(0), ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO),
        target_type));

    return castedConstant;
}

std::shared_ptr<Node> NetworkHelper::fold_fake_quantize(const std::shared_ptr<opset1::FakeQuantize>& fq) {
    return foldFakeQuantize(fq, false, false);
}

std::shared_ptr<Node> NetworkHelper::fold_fake_quantize(
        const std::shared_ptr<opset1::FakeQuantize>& fq,
        const bool roundValues,
        const int outChannelsShapeIndex) {
    return foldFakeQuantize(fq, roundValues, true, outChannelsShapeIndex);
}

FakeQuantizeDequantization NetworkHelper::foldDequantization(const std::shared_ptr<Node>& node, const size_t branchIndex, const bool inPlace) {
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(node, branchIndex, inPlace);
    if (dequantization.empty() || (dequantization.multiply == nullptr)) {
        return dequantization;
    }

    if (dequantization.convert != nullptr) {
        const std::shared_ptr<Node> result = foldConvert(dequantization.data, dequantization.convert->get_element_type());
        if (is_type<opset1::Constant>(result)) {
            if (inPlace) {
                copyInfo(dequantization.convert, result);
            }
            replace_node(dequantization.convert, result);
            dequantization = NetworkHelper::getDequantization(node, branchIndex, inPlace);
        }
    }

    if (dequantization.subtract != nullptr) {
        if (dequantization.subtract->input(0).get_element_type() != dequantization.subtract->input(1).get_element_type()) {
            return dequantization;
        }

        if (dequantization.subtractConvert != nullptr) {
            const auto convertionResult = foldConvert(
                dequantization.subtractConstant,
                dequantization.subtractConvert->get_element_type());
            if (is_type<opset1::Constant>(convertionResult)) {
                replace_node(dequantization.subtractConvert, convertionResult);
                dequantization = NetworkHelper::getDequantization(node, branchIndex, inPlace);
            }
        }

        const std::shared_ptr<Node> result = fold<opset1::Subtract>(
            dequantization.subtract->get_input_node_shared_ptr(0),
            dequantization.subtract->get_input_node_shared_ptr(1));
        if (is_type<opset1::Constant>(result)) {
            if (inPlace) {
                copyInfo(dequantization.subtract, result);
            }
            replace_node(dequantization.subtract, result);
            dequantization = NetworkHelper::getDequantization(node, branchIndex, inPlace);
        } else {
            return dequantization;
        }
    }

    if (dequantization.multiply != nullptr) {
        if (dequantization.multiply->input(0).get_element_type() != dequantization.multiply->input(1).get_element_type()) {
            return dequantization;
        }

        std::shared_ptr<Node> result = fold<opset1::Multiply>(
                dequantization.multiply->get_input_node_shared_ptr(0),
                dequantization.multiply->get_input_node_shared_ptr(1));
        if (!is_type<opset1::Constant>(result)) {
            return dequantization;
        }
        if (dequantization.multiply->get_output_element_type(0) != result->get_element_type()) {
            result = foldConvert(result, dequantization.multiply->get_output_element_type(0));
        }
        if (inPlace) {
            copyInfo(dequantization.multiply, result);
        }
        replace_node(dequantization.multiply, result);
        dequantization = NetworkHelper::getDequantization(node, branchIndex, inPlace);
    }


    return dequantization;
}

std::shared_ptr<ngraph::Node> NetworkHelper::separateInStandaloneBranch(std::shared_ptr<ngraph::Node> node) {
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(node);
    if (dequantization.isShared()) {
        Output<Node> parent = dequantization.data;
        if (dequantization.convert != nullptr) {
            parent = dequantization.convert->clone_with_new_inputs({ parent });
            parent.get_node_shared_ptr()->set_friendly_name(parent.get_node_shared_ptr()->get_name() + "_new");
        }

        if (dequantization.subtract != nullptr) {
            const auto parentOnWeights = dequantization.subtract->get_input_node_shared_ptr(1);
            const std::vector<Input<Node>> inputs = parentOnWeights->inputs();
            OutputVector outputs;
            outputs.reserve(inputs.size());
            for (const auto& input : inputs) {
                outputs.push_back(input.get_source_output());
            }

            parent = dequantization.subtract->clone_with_new_inputs({parent, parentOnWeights->clone_with_new_inputs(outputs) });
            parent.get_node_shared_ptr()->set_friendly_name(parent.get_node_shared_ptr()->get_name() + "_new");
        }

        if (dequantization.multiply != nullptr) {
            parent = dequantization.multiply->clone_with_new_inputs({
                parent,
                dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) });
            parent.get_node_shared_ptr()->set_friendly_name(parent.get_node_shared_ptr()->get_name() + "_new");
        }

        std::vector<Output<Node>> inputs = node->input_values();
        const size_t inputIndex = NetworkHelper::getChildInputIndex(dequantization.multiply, node);
        inputs[inputIndex] = parent;
        const std::shared_ptr<Node> newNode = node->clone_with_new_inputs(inputs);

        replace_node(node, newNode);
        newNode->set_friendly_name(node->get_friendly_name());

        return newNode;
    }

    return node;
}

std::shared_ptr<opset1::FakeQuantize> NetworkHelper::fuseConvert(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) {
    const Output<Node> output = fakeQuantize->output(0);
    const auto targetInputs = output.get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return fakeQuantize;
    }

    Node* node = targetInputs.begin()->get_node();
    if (!is_type<opset1::Convert>(node) ||
        // TODO: LPT: avoid precision hardcode: to separate method: isSupportedPrecision
        ((node->get_output_element_type(0) != element::u8) && (node->get_output_element_type(0) != element::i8))) {
        return fakeQuantize;
    }


    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset1::FakeQuantize>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32, element::f32, element::f32, element::f32 },
        std::vector<ngraph::element::Type>{},
        ngraph::op::TemporaryReplaceOutputType(fakeQuantize->get_input_node_shared_ptr(0), element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(fakeQuantize->get_input_node_shared_ptr(1), element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(fakeQuantize->get_input_node_shared_ptr(2), element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(fakeQuantize->get_input_node_shared_ptr(3), element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(fakeQuantize->get_input_node_shared_ptr(4), element::f32).get(),
        fakeQuantize->get_levels());
    NetworkHelper::setOutDataPrecisionForTypeRelaxed(newFakeQuantize, node->get_output_element_type(0));
    replace_node(node->shared_from_this(), newFakeQuantize);
    newFakeQuantize->set_friendly_name(fakeQuantize->get_friendly_name());
    return newFakeQuantize;
}

std::shared_ptr<Node> NetworkHelper::foldFakeQuantize(
    const std::shared_ptr<opset1::FakeQuantize>& fq,
    const bool roundValuesArg,
    const bool roundValuesWasSet,
    const int outChannelsShapeIndex) {
    if (is_type<opset1::Constant>(fq->get_input_node_shared_ptr(0)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(1)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(2)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(3)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(4)) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(1)), 0.f) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(2)), 254.f) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(3)), -127.f) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(4)), 127.f)) {
        const auto type1 = fq->input_value(0).get_element_type();
        const auto type2 = fq->input_value(3).get_element_type();
        if (type1.is_real() && type2.is_real()) {
            return fold<opset1::Add>(fq->input_value(0), fq->input_value(3));
        }
        if (type1.is_real() && !type2.is_real()) {
            return fold<opset1::Add>(
                fq->input_value(0),
                foldConvert(fq->input_value(3), type1));
        }
        if (!type1.is_real() && type2.is_real()) {
            return fold<opset1::Add>(
                foldConvert(fq->input_value(0), type2),
                fq->input_value(3));
        }
        return fold<opset1::Add>(
            foldConvert(fq->input_value(0), element::f32),
            foldConvert(fq->input_value(3), element::f32));
    }

    auto constant = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(0));

    if (constant) {
        const bool roundValues = roundValuesWasSet ? roundValuesArg : fq->output(0).get_element_type().is_integral();

        Shape constShape = fq->get_output_shape(0);
        if (constShape.empty() || constShape.size() > 5lu) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected dimensions count " << constShape.size();
        }
        if (outChannelsShapeIndex != 0 && outChannelsShapeIndex != 1) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected outChannelsShapeIndex " << outChannelsShapeIndex;
        }

        size_t OC;
        size_t IC;
        // OIDHW or IODHW
        if (constShape.size() == 1) {
            OC = constShape[0];
            IC = 1;
        } else {
            OC = constShape[outChannelsShapeIndex];
            IC = constShape[outChannelsShapeIndex == 0 ? 1 : 0];
        }
        const size_t D = constShape.size() > 4lu ? constShape[constShape.size() - 3] : 1;
        const size_t H = constShape.size() > 2lu ? constShape.size() == 3lu ? constShape[2] : constShape[constShape.size() - 2] : 1;
        const size_t W = constShape.size() > 3lu ? constShape[constShape.size() - 1] : 1;

        const auto inputLowValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(1))->cast_vector<float>();
        const auto inputHighValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(2))->cast_vector<float>();
        const auto outputLowValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(3))->cast_vector<float>();
        const auto outputHighValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(4))->cast_vector<float>();

        const size_t inputLowSize = inputLowValues.size();
        const size_t inputHighSize = inputHighValues.size();
        const size_t outputLowSize = outputLowValues.size();
        const size_t outputHighSize = outputHighValues.size();

        const bool isInputLowBroadcasted = inputLowSize != OC;
        if ((inputLowSize != 1) && (inputLowSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected input low values count " << inputLowSize << " for " << OC << " channels";
        }
        const bool isInputHighBroadcasted = inputHighSize != OC;
        if ((inputHighSize != 1) && (inputHighSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected input high values count " << inputHighSize << " for " << OC << " channels";
        }
        const bool isOutputLowBroadcasted = outputLowSize != OC;
        if ((outputLowSize != 1) && (outputLowSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected output low values count " << outputLowSize << " for " << OC << " channels";
        }
        const bool isOutputHighBroadcasted = outputHighSize != OC;
        if ((outputHighSize != 1) && (outputHighSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected output high values count " << outputHighSize << " for " << OC << " channels";
        }

        auto levels_1 = fq->get_levels() - 1.f;

        const size_t DHW = D * H * W;
        const size_t IDHW = outChannelsShapeIndex == 0 ? IC * D * H * W : OC * D * H * W;

        const auto values = constant->cast_vector<float>();
        std::vector<float> quantizedValues(OC * IC * D * H * W);

        for (size_t oc = 0; oc < OC; ++oc) {
            const float inputLow = inputLowValues[isInputLowBroadcasted ? 0 : oc];
            const float inputHigh = inputHighValues[isInputHighBroadcasted ? 0 : oc];
            const float outputLow = outputLowValues[isOutputLowBroadcasted ? 0 : oc];
            const float outputHigh = outputHighValues[isOutputHighBroadcasted ? 0 : oc];
            for (size_t ic = 0; ic < IC; ++ic) {
                for (size_t iidx = 0; iidx < DHW; ++iidx) {
                    size_t idx;
                    if (outChannelsShapeIndex == 0) {
                        idx = oc * IDHW + ic * DHW + iidx;
                    } else {
                        idx = ic * IDHW + oc * DHW + iidx;
                    }

                    if (values[idx] <= inputLow) {
                        quantizedValues[idx] = roundValues ? std::roundf(outputLow) : outputLow;
                    } else if (values[idx] > inputHigh) {
                        quantizedValues[idx] = roundValues ? std::roundf(outputHigh) : outputHigh;
                    } else {
                        const float value = std::roundf((values[idx] - inputLow) / (inputHigh - inputLow) * levels_1) /
                            levels_1 * (outputHigh - outputLow) + outputLow;
                        quantizedValues[idx] = roundValues ? std::roundf(value) : value;
                    }
                }
            }
        }

        return std::make_shared<opset1::Constant>(fq->get_output_element_type(0), constShape, quantizedValues);
    }

    return fq;
}

std::shared_ptr<opset1::FakeQuantize> NetworkHelper::composeFakeQuantize(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) {
    std::shared_ptr<Node> parent = fakeQuantize;
    auto targetInputs = parent->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return nullptr;
    }
    if (is_type<opset1::Convert>(targetInputs.begin()->get_node())) {
        parent = targetInputs.begin()->get_node()->shared_from_this();
    }

    targetInputs = parent->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return nullptr;
    }
    if (is_type<opset1::Subtract>(targetInputs.begin()->get_node())) {
        parent = targetInputs.begin()->get_node()->shared_from_this();
    }

    targetInputs = parent->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return nullptr;
    }
    if (is_type<opset1::Multiply>(targetInputs.begin()->get_node())) {
        parent = targetInputs.begin()->get_node()->shared_from_this();
    }

    const std::shared_ptr<Node> prev = parent;
    parent = parent->output(0).get_target_inputs().begin()->get_node()->shared_from_this();

    const size_t index = NetworkHelper::getChildInputIndex(prev, parent);
    const FakeQuantizeDequantization dequantization = getDequantization(parent, index);
    if (dequantization.empty()) {
        return nullptr;
    }

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = fakeQuantize;

    if (dequantization.convert != nullptr) {
        const std::shared_ptr<opset1::FakeQuantize> replacement = std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
            newFakeQuantize->input_value(0),
            newFakeQuantize->input_value(1),
            newFakeQuantize->input_value(2),
            newFakeQuantize->input_value(3),
            newFakeQuantize->input_value(4),
            newFakeQuantize->get_levels(),
            newFakeQuantize->get_auto_broadcast());
        replace_node(dequantization.convert, replacement);
        replacement->set_friendly_name(newFakeQuantize->get_friendly_name());
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, dequantization.convert->output(0).get_element_type());
        newFakeQuantize = replacement;
    }

    if (dequantization.subtract != nullptr) {
        const auto subtractValue = (dequantization.subtractConvert == nullptr) ?
            dequantization.subtractConstant :
            foldConvert(dequantization.subtractConstant, dequantization.subtractConvert->output(0).get_element_type());

        const std::shared_ptr<opset1::FakeQuantize> replacement = std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
            newFakeQuantize->input_value(0),
            newFakeQuantize->input_value(1),
            newFakeQuantize->input_value(2),
            fold<opset1::Subtract>(newFakeQuantize->get_input_node_shared_ptr(3), subtractValue),
            fold<opset1::Subtract>(newFakeQuantize->get_input_node_shared_ptr(4), subtractValue),
            newFakeQuantize->get_levels(),
            newFakeQuantize->get_auto_broadcast());
        replace_node(dequantization.subtract, replacement);
        replacement->set_friendly_name(newFakeQuantize->get_friendly_name());
        newFakeQuantize = replacement;
    }

    if (dequantization.multiply != nullptr) {
        // multiply different precision constants (value1 & value2) and convert result to first argument precision (value1)
        auto multiply = [](
            const std::shared_ptr<ngraph::Node>& value1,
            const std::shared_ptr<ngraph::Node>& value2) -> std::shared_ptr<ngraph::Node> {
            const ngraph::element::Type precision1 = value1->output(0).get_element_type();
            const ngraph::element::Type precision2 = value2->output(0).get_element_type();
            // 1) precision1 & precision2 are not equal but similar
            // 2) precision2 >= precision1
            assert((precision2.is_real() == precision1.is_real()) && (precision2.bitwidth() >= precision1.bitwidth()));

            auto output = fold<opset1::Multiply>(
                precision2 != precision1 ? foldConvert(value1, precision2) : value1,
                value2);

            if (output->output(0).get_element_type() != precision1) {
                output = foldConvert(output, precision1);
            }

            return output;
        };

        const std::shared_ptr<opset1::FakeQuantize> replacement = std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
            newFakeQuantize->input_value(0ul),
            newFakeQuantize->input_value(1ul),
            newFakeQuantize->input_value(2ul),
            multiply(newFakeQuantize->get_input_node_shared_ptr(3ul), dequantization.multiplyConstant),
            multiply(newFakeQuantize->get_input_node_shared_ptr(4ul), dequantization.multiplyConstant),
            newFakeQuantize->get_levels(),
            newFakeQuantize->get_auto_broadcast());

        replace_node(dequantization.multiply, replacement);
        replacement->set_friendly_name(newFakeQuantize->get_friendly_name());
        newFakeQuantize = replacement;
    }

    return newFakeQuantize;
}

// Decompose FakeQuantize to FakeQuantize with output integer limits (quantize), dequatized MultiplyAdd
// To align types the resulting sequence is FakeQuantize -> Convert -> Convert -> MultiplyAdd
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> NetworkHelper::decomposeFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    const element::Type precision,
    const float min,
    const float max,
    const bool hasZeroPoint,
    const bool updatePrecision,
    const element::Type deqPrecision,
    const size_t outChannelsShapeIndex) {
    using std::make_shared;

    const auto outputLow = fq->input_value(3);
    const auto outputHigh = fq->input_value(4);

    std::vector<float> outputLowValues = as_type_ptr<opset1::Constant>(outputLow.get_node_shared_ptr())->cast_vector<float>();
    std::vector<float> outputHighValues = as_type_ptr<opset1::Constant>(outputHigh.get_node_shared_ptr())->cast_vector<float>();
    size_t outputSize = outputLowValues.size();
    std::vector<float> minValues(outputSize, min);
    std::vector<float> maxValues(outputSize, max);
    std::vector<float> shifts(outputSize, 0.f);
    std::vector<float> scales(outputSize);

    for (size_t i = 0; i < outputSize; ++i) {
        if (outputHighValues[i] != outputLowValues[i]) {
            shifts[i] = (min*outputHighValues[i] - max*outputLowValues[i]) / (outputHighValues[i] - outputLowValues[i]);
            scales[i] = (outputHighValues[i] - outputLowValues[i]) / (max - min);
            if (shifts[i] == -0.f) {
                shifts[i] = 0.f;
            }
        } else {
            scales[i] = outputHighValues[i];
            minValues[i] = 1.f;
            maxValues[i] = 1.f;
        }
    }

    std::shared_ptr<Node> shift = hasZeroPoint ?
        std::make_shared<opset1::Constant>(deqPrecision, outputLow.get_shape(), shifts) :
        nullptr;
    std::shared_ptr<Node> scale = std::make_shared<opset1::Constant>(element::f32, outputLow.get_shape(), scales);

    auto newMin = make_shared<opset1::Constant>(outputLow.get_element_type(), outputLow.get_shape(), minValues);
    auto newMax = make_shared<opset1::Constant>(outputLow.get_element_type(), outputLow.get_shape(), maxValues);

    if (isScalarLike(newMin)) {
        newMin = toScalar(newMin);
    }
    if (isScalarLike(newMax)) {
        newMax = toScalar(newMax);
    }

    {
        static const float minQuantizationScale = 1e-32f;
        static const float maxQuantizationScale = 1e32f;

        auto scaleValues = scales;
        bool wasChanged = false;
        for (size_t i = 0; i < scaleValues.size(); ++i) {
            const float scale = scaleValues[i];
            if (fabs(scale) < minQuantizationScale) {
                scaleValues[i] = minQuantizationScale;
                wasChanged = true;
            } else if (fabs(scale) > maxQuantizationScale) {
                scaleValues[i] = scale > 0.f ? maxQuantizationScale : -maxQuantizationScale;
                wasChanged = true;
            }
        }

        if (wasChanged) {
            scale = std::make_shared<opset1::Constant>(scale->output(0).get_element_type(), scale->output(0).get_shape(), scaleValues);
        }
    }

    if ((shift != nullptr) && isZero(as_type_ptr<opset1::Constant>(shift))) {
        shift = nullptr;
    }

    // Build a substitution sub-graph:

    std::shared_ptr<ngraph::Node> newFQ = fold_fake_quantize(
        std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
            fq->input_value(0),
            fq->input_value(1),
            fq->input_value(2),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast()),
        true,
        outChannelsShapeIndex);
    NetworkHelper::copyInfo(fq, newFQ);

    std::shared_ptr<ngraph::Node> convert2;
    if (updatePrecision) {
        std::shared_ptr<Node> convert;
        std::shared_ptr<opset1::Constant> newFqConstant = as_type_ptr<opset1::Constant>(newFQ);

        if (is_type<opset1::Constant>(newFQ)) {
            convert = foldConvert(newFQ, precision);
        } else if (is_type<opset1::FakeQuantize>(newFQ)) {
            newFQ = setOutDataPrecision(as_type_ptr<opset1::FakeQuantize>(newFQ), precision);
            convert = newFQ;
        } else {
            THROW_IE_LPT_EXCEPTION(*newFQ) << "unexpected operation type";
        }

        convert2 = std::make_shared<DequantizationConvert>(convert, element::f32);
        convert2->set_friendly_name(convert->get_friendly_name() + "/DequantizationConvert");
        ngraph::copy_runtime_info({ newFQ, convert2 }, convert2);
    } else {
        if (newFQ->get_output_element_type(0) != element::f32) {
            convert2 = std::make_shared<DequantizationConvert>(newFQ, element::f32);
            convert2->set_friendly_name(newFQ->get_friendly_name() + "/DequantizationConvert");
            ngraph::copy_runtime_info({ newFQ, convert2 }, convert2);
        }
    }

    // TODO: why type relaxed?
    const std::shared_ptr<ngraph::Node> sub = shift == nullptr ?
        nullptr :
        std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(convert2 == nullptr ? newFQ : convert2, shift);
    if (sub != nullptr) {
        sub->set_friendly_name(newFQ->get_friendly_name() + "/DequantizationSubtract");
        ngraph::copy_runtime_info({ newFQ, sub }, sub);
    }

    const auto dequantize =
        std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ fq->get_output_element_type(0) },
            ngraph::op::TemporaryReplaceOutputType(sub == nullptr ? (convert2 == nullptr ? newFQ : convert2) : sub, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(scale, element::f32).get());
    dequantize->set_friendly_name(newFQ->get_friendly_name() + "/DequantizationMultiply");
    ngraph::copy_runtime_info({ newFQ, dequantize }, dequantize);

    replace_node(fq, dequantize);

    return std::make_tuple(newFQ, dequantize);
}

std::shared_ptr<opset1::FakeQuantize> NetworkHelper::updateFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max) {
    auto newMin = std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newMax = std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    std::shared_ptr<opset1::FakeQuantize> newFQ = std::make_shared<ngraph::op::TypeRelaxed<opset1::FakeQuantize>>(
            fq->input_value(0),
            fq->input_value(1),
            fq->input_value(2),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast());

    NetworkHelper::setOutDataPrecision(newFQ, precision);
    replace_node(fq, newFQ);

    newFQ->set_friendly_name(fq->get_friendly_name());
    return newFQ;
}

FakeQuantizeDequantization NetworkHelper::makeDequantization(
    const float dequantizationMul,
    const float dequantizationSub,
    const ngraph::element::Type originalPrecision,
    const ngraph::Shape dataNodeOutputShape,
    element::Type precision,
    const ngraph::element::Type deqPrecision) {
    // TODO: we create input here! we really need it here?
    const std::shared_ptr<opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(precision, dataNodeOutputShape);
    std::shared_ptr<ngraph::Node> parent = input;

    std::shared_ptr<DequantizationConvert> convert;
    if (precision == deqPrecision) {
        convert = nullptr;
    } else {
        convert = std::make_shared<DequantizationConvert>(
            input,
            deqPrecision);
        parent = convert;
    }

    std::shared_ptr<DequantizationSubtract> subtract;
    std::shared_ptr<opset1::Constant> subtractConstant;
    if (std::abs(dequantizationSub) > 1e-6) {
        subtract = std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(deqPrecision, ngraph::Shape({}), std::vector<float>({ dequantizationSub })));
        subtract->set_output_type(0, deqPrecision, subtract->get_output_partial_shape(0));
        parent = subtract;
    }

    // mandatory
    auto multiplyConstant = std::make_shared<ngraph::opset1::Constant>(deqPrecision, ngraph::Shape({}), std::vector<float>({ dequantizationMul }));
    auto multiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
        DequantizationMultiply(parent, multiplyConstant),
        originalPrecision);

    return FakeQuantizeDequantization(input, convert, subtract, nullptr, subtractConstant, multiply, multiplyConstant);
}

FakeQuantizeDequantization NetworkHelper::createDequantizationFromFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max,
    const bool hasZeroPoint,
    const bool updatePrecision,
    const element::Type deqPrecision) {
    using std::make_shared;

    const ngraph::element::Type_t fqPrecision = fq->get_output_element_type(0);
    auto newMin = make_shared<opset1::Constant>(fqPrecision, Shape{}, min);
    auto newMax = make_shared<opset1::Constant>(fqPrecision, Shape{}, max);

    auto outputLow = fq->input_value(3);
    auto outputHigh = fq->input_value(4);

    // TODO: threshold values have to used here to avoid shifts

    const std::shared_ptr<opset1::Constant> scale = as_type_ptr<opset1::Constant>(foldConvert(fold<opset1::Divide>(
        fold<opset1::Subtract>(outputHigh, outputLow),
        fold<opset1::Subtract>(newMax, newMin)), deqPrecision));
    assert(scale != nullptr);

    std::shared_ptr<opset1::Constant> shift = hasZeroPoint ?
        as_type_ptr<opset1::Constant>(foldConvert(fold<opset1::Divide>(
            fold<opset1::Subtract>(fold<opset1::Multiply>(newMin, outputHigh), fold<opset1::Multiply>(newMax, outputLow)),
            fold<opset1::Subtract>(outputHigh, outputLow)), deqPrecision)) :
        nullptr;
    assert((!hasZeroPoint) || (hasZeroPoint && shift != nullptr));

    if (shift != nullptr) {
        std::shared_ptr<opset1::Constant> shiftConst = as_type_ptr<opset1::Constant>(shift);
        if (isScalarLike(shiftConst)) {
            auto scalar = toScalar(shiftConst);
            if (op::util::constantIsEqualTo(scalar, 0)) {
                shift = nullptr;
            }
        }
    }

    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        updatePrecision ? precision : fq->get_output_element_type(0),
        fq->get_output_shape(0));
    std::shared_ptr<ngraph::Node> parent = input;

    std::shared_ptr<ngraph::opset1::Convert> convert;
    if (updatePrecision || (parent->output(0).get_element_type() != deqPrecision)) {
        convert = std::make_shared<DequantizationConvert>(parent, deqPrecision);
        parent = convert;
    } else {
        convert = nullptr;
    }

    std::shared_ptr<ngraph::opset1::Subtract> subtract;
    if (shift != nullptr) {
        subtract = make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(parent, shift);
        subtract->set_output_type(0, deqPrecision, subtract->get_output_partial_shape(0));
        parent = subtract;
    } else {
        subtract = nullptr;
    }
    const std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<DequantizationMultiply>(parent, scale);
    multiply->set_output_type(0, fq->get_output_element_type(0), multiply->get_output_partial_shape(0));

    return FakeQuantizeDequantization(fq, convert, subtract, nullptr, shift, multiply, scale);
}

bool NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(const std::shared_ptr<const ngraph::Node>& node) {
    if (!is_type<opset1::Subtract>(node)) {
        return false;
    }

    const auto targetInputs = node->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return false;
    }

    const auto multiply = targetInputs.begin()->get_node()->shared_from_this();
    return areQuantizeAndDequantizeSupportedForMultiply(multiply);
}

bool NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(const std::shared_ptr<const ngraph::Node>& node) {
    if (!is_type<opset1::Multiply>(node)) {
        return false;
    }

    const std::shared_ptr<ngraph::Node> multiply = const_cast<ngraph::Node*>(node.get())->shared_from_this();
    const auto dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(multiply, 0, true);
    if (dequantization.empty()) {
        return false;
    }

    const auto dataNode = dequantization.data.get_node();
    if (is_type<opset1::Convert>(dataNode)) {
        const auto quantize = as_type_ptr<opset1::FakeQuantize>(dataNode->get_input_node_shared_ptr(0));
        if (quantize == nullptr) {
            return false;
        }

        return NetworkHelper::isQuantizeSupported(quantize);
    } else if (is_type<opset1::Constant>(dataNode)) {
        return true;
    }

    return false;
}

bool NetworkHelper::isQuantizeSupported(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) {
    return QuantizationDetails::outputLayoutIsSupported(fakeQuantize) && QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels());
}

FakeQuantizeDequantization NetworkHelper::getDequantization(const std::shared_ptr<Node>& node, const size_t parentIndex, const bool inPlace) {
    auto getDataIndex = [](const std::shared_ptr<ngraph::Node>& node) {
        if (is_type<opset1::Constant>(node->get_input_node_ptr(1))) {
            return 0ul;
        }

        if (is_type<opset1::Convert>(node->get_input_node_ptr(1)) && is_type<opset1::Constant>(node->get_input_node_ptr(1)->get_input_node_ptr(0))) {
            return 0ul;
        }

        if (is_type<opset1::Convert>(node->get_input_node_ptr(0)) && is_type<opset1::Constant>(node->get_input_node_ptr(0)->get_input_node_ptr(0))) {
            return 1ul;
        }

        return 1ul;
    };

    Output<Node> dataNode = inPlace ? node->output(0) : node->input_value(parentIndex);

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = as_type_ptr<ngraph::opset1::Multiply>(dataNode.get_node_shared_ptr());
    std::shared_ptr<opset1::Constant> multiplyConstant;
    if (multiply != nullptr) {
        if (!FakeQuantizeDequantization::checkShape(multiply)) {
            return FakeQuantizeDequantization();
        }

        FakeQuantizeDequantization::fillDequantizationParams(multiply, multiplyConstant);
        if (multiplyConstant == nullptr) {
            return FakeQuantizeDequantization();
        }
        dataNode = multiply->get_input_source_output(getDataIndex(multiply));
    }

    const std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<ngraph::opset1::Subtract>(dataNode.get_node_shared_ptr());
    std::shared_ptr<opset1::Convert> subtractConvert;
    std::shared_ptr<opset1::Constant> subtractConstant;
    if (subtract != nullptr) {
        if (!FakeQuantizeDequantization::checkShape(subtract)) {
            return FakeQuantizeDequantization(dataNode, nullptr, nullptr, nullptr, nullptr, multiply, multiplyConstant);
        }

        FakeQuantizeDequantization::fillDequantizationParams(subtract, subtractConvert, subtractConstant);
        if (subtractConstant == nullptr) {
            return FakeQuantizeDequantization(dataNode, nullptr, nullptr, nullptr, nullptr, multiply, multiplyConstant);
        }
        dataNode = subtract->get_input_source_output(getDataIndex(subtract));
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode.get_node_shared_ptr());
    if (convert != nullptr) {
        if ((convert->input(0).get_element_type() != element::i8) && (convert->input(0).get_element_type() != element::u8) &&
            (convert->output(0).get_element_type() != element::f32)) {
            return FakeQuantizeDequantization(dataNode, nullptr, subtract, subtractConvert, subtractConstant, multiply, multiplyConstant);
        }
        dataNode = convert->get_input_source_output(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, subtractConvert, subtractConstant, multiply, multiplyConstant);
}

FakeQuantizeDequantization NetworkHelper::getDequantizationBelow(const std::shared_ptr<Node>& node) {
    const Output<Node> dataNode = node->output(0);
    std::shared_ptr<Node> lastNode = dataNode.get_target_inputs().begin()->get_node()->shared_from_this();

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(lastNode);
    if (convert != nullptr) {
        if ((convert->input(0).get_element_type() != element::i8) && (convert->input(0).get_element_type() != element::u8) &&
            (convert->output(0).get_element_type() != element::f32)) {
            return FakeQuantizeDequantization();
        }

        const auto& inputs = lastNode->output(0).get_target_inputs();
        if (inputs.size() != 1ul) {
            return FakeQuantizeDequantization();
        }
        lastNode = inputs.begin()->get_node()->shared_from_this();
    }

    const std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<ngraph::opset1::Subtract>(lastNode);
    std::shared_ptr<opset1::Convert> subtractConvert;
    std::shared_ptr<opset1::Constant> subtractConstant;
    if (subtract != nullptr) {
        FakeQuantizeDequantization::fillDequantizationParams(subtract, subtractConvert, subtractConstant);
        if (subtractConstant == nullptr) {
            return FakeQuantizeDequantization();
        }

        const auto& inputs = lastNode->output(0).get_target_inputs();
        if (inputs.size() != 1ul) {
            return FakeQuantizeDequantization();
        }
        lastNode = inputs.begin()->get_node()->shared_from_this();
    }

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = as_type_ptr<ngraph::opset1::Multiply>(lastNode);
    std::shared_ptr<opset1::Constant> multiplyConstant;
    if (multiply != nullptr) {
        FakeQuantizeDequantization::fillDequantizationParams(multiply, multiplyConstant);
        if (multiplyConstant == nullptr) {
            return FakeQuantizeDequantization();
        }
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, subtractConvert, subtractConstant, multiply, multiplyConstant);
}

FakeQuantizeDequantization NetworkHelper::normalizeDequantization(FakeQuantizeDequantization dequantization) {
    if (dequantization.empty()) {
        return dequantization;
    }
    if (dequantization.multiply != nullptr && as_type_ptr<ngraph::opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(0))) {
        std::shared_ptr<Node> leftParent = dequantization.multiply->get_input_node_shared_ptr(0);
        std::shared_ptr<Node> rightParent = dequantization.multiply->get_input_node_shared_ptr(1);
        std::shared_ptr<opset1::Multiply> normalized_multiply = as_type_ptr<opset1::Multiply>(
                dequantization.multiply->clone_with_new_inputs({rightParent, leftParent}));
        replace_node(dequantization.multiply, normalized_multiply);
        dequantization.multiply = normalized_multiply;
    }
    if (dequantization.subtract != nullptr && as_type_ptr<ngraph::opset1::Constant>(dequantization.subtract->get_input_node_shared_ptr(0))) {
        std::shared_ptr<Node> leftParent = dequantization.subtract->get_input_node_shared_ptr(0);
        std::shared_ptr<Node> rightParent = dequantization.subtract->get_input_node_shared_ptr(1);
        std::shared_ptr<opset1::Subtract> normalized_subtract = as_type_ptr<opset1::Subtract>(
                dequantization.subtract->clone_with_new_inputs({rightParent, leftParent}));
        replace_node(dequantization.subtract, normalized_subtract);
        dequantization.subtract = normalized_subtract;
    }
    return dequantization;
}

std::shared_ptr<opset1::Constant> NetworkHelper::normalizeDequantizationShape(const std::shared_ptr<Node>& eltwise) {
    const size_t constantIdx = getConstantInputIndex(eltwise);
    const auto constant = as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(constantIdx));

    const auto getConstWithNormalizeShape = [](
        const std::shared_ptr<Node>& eltwise,
        const std::shared_ptr<opset1::Constant>& constant) {
        const auto constantShape = constant->get_shape();
        if (constantShape.empty()) {
            return constant;
        }

        const auto eltwiseShape = eltwise->get_output_shape(0);
        if (constantShape.size() < eltwiseShape.size()) {
            Shape unsqueezeConstantShape(eltwiseShape.size() - constantShape.size());
            std::iota(unsqueezeConstantShape.begin(), unsqueezeConstantShape.end(), 0ul);

            const auto newConstant = fold<opset1::Unsqueeze>(
                constant,
                op::Constant::create(element::i32, Shape{ unsqueezeConstantShape.size() }, unsqueezeConstantShape));

            return as_type_ptr<opset1::Constant>(newConstant);
        } else {
            return constant;
        }
    };

    const auto normalizedConstant = getConstWithNormalizeShape(eltwise, constant);
    replace_node(constant, normalizedConstant);
    copy_runtime_info(constant, normalizedConstant);

    return normalizedConstant;
}

FakeQuantizeDequantizationValues NetworkHelper::createEmptyValues(const FakeQuantizeDequantization& dequantization) {
    std::shared_ptr<Node> parent = dequantization.convert ? dequantization.convert : dequantization.data.get_node_shared_ptr();

    std::shared_ptr<Node> multiply1Const = dequantization.multiply ?
        dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 1.f }));

    std::shared_ptr<Node> subtract1Const = dequantization.subtract ?
        (dequantization.subtractConvert == nullptr ?
            dequantization.subtractConstant->clone_with_new_inputs({}) :
            foldConvert(dequantization.subtractConstant, dequantization.subtractConvert->get_element_type())) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 0.f }));

    subtract1Const->set_output_type(0, multiply1Const->get_output_element_type(0), subtract1Const->get_output_partial_shape(0));

    return FakeQuantizeDequantizationValues(subtract1Const, multiply1Const);
}

bool NetworkHelper::isZeroConst(const std::shared_ptr<Node>& node) {
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(node);

    if (constant == nullptr)
        return false;

    if (NetworkHelper::isScalarLike(constant)) {
        auto scalar = NetworkHelper::toScalar(constant);
        if (op::util::constantIsEqualTo(scalar, 0)) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

std::shared_ptr<Node> NetworkHelper::optimizeSubtract(std::shared_ptr<opset1::Subtract> subtract) {
    auto convertOnSubtract = subtract->input_value(0).get_node_shared_ptr();
    if (as_type_ptr<opset1::Convert>(convertOnSubtract) == nullptr) {
        return subtract;
    }

    // TODO: replace assert to condition and omit conversion part if there is no convert
    // TODO: also check convertInputType to understand if we really want to propagate type
    assert(as_type_ptr<opset1::Convert>(convertOnSubtract));
    const element::Type convertInputType = convertOnSubtract->get_input_element_type(0);
    const element::Type convertOutputType = convertOnSubtract->get_output_element_type(0);

    if (!convertOutputType.is_real()) {
        return subtract;
    }

    auto data = convertOnSubtract->input_value(0);
    const auto subtractParent = subtract->get_input_node_shared_ptr(1);
    if (is_type<opset1::Constant>(subtractParent)) {
        std::shared_ptr<Node> replacement;

        auto shift = subtract->input_value(1).get_node_shared_ptr();
        auto roundedShift = NetworkHelper::round(shift, convertInputType);

        if (isScalarLike(roundedShift)) {
            roundedShift = toScalar(roundedShift);
            if (op::util::constantIsEqualTo(roundedShift, 0)) {
                replace_node(subtract, convertOnSubtract->get_input_node_shared_ptr(0));
                roundedShift = nullptr;
            }
        }

        if (roundedShift) {
            NetworkHelper::copyInfo(shift, roundedShift);

            // Propagate convertInputType down
            replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(data, roundedShift);
            NetworkHelper::copyInfo(subtract, replacement);
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, convertOutputType);
            replace_node(subtract, replacement);
        }

        return replacement;
    } else if (is_type<opset1::Convert>(subtractParent) && is_type<opset1::Constant>(subtractParent->get_input_node_shared_ptr(0))) {
        auto replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(data, subtractParent->get_input_node_shared_ptr(0));
        NetworkHelper::copyInfo(subtract, replacement);
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, convertOutputType);
        replace_node(subtract, replacement);
        return replacement;
    }

    return subtract;
}

NetworkHelper::InsertDequantizationResult NetworkHelper::moveDequantizationAfter(
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updatePrecision,
    const bool moveSubtract) {
    assert(
        (NetworkHelper::getDequantization(operation).subtractConstant == nullptr) ||
        (NetworkHelper::getDequantization(operation).subtractConstant.get() == dequantization.subtractConstant.get()));

    assert(
        (NetworkHelper::getDequantization(operation).multiplyConstant == nullptr) ||
        (NetworkHelper::getDequantization(operation).multiplyConstant.get() == dequantization.multiplyConstant.get()));

    std::vector<Output<Node>> inputs(operation->get_input_size());
    for (size_t i = 0; i < operation->get_input_size(); ++i) {
        inputs[i] = operation->get_input_node_shared_ptr(i);
    }

    const size_t dequantizationIndex = getChildInputIndex(dequantization.multiply, operation);
    inputs[dequantizationIndex] = moveSubtract ?
        dequantization.data :
        (dequantization.subtract == nullptr ? dequantization.data : dequantization.subtract);

    const std::shared_ptr<ngraph::Node> newOperation = operation->clone_with_new_inputs(inputs);
    newOperation->set_friendly_name(operation->get_friendly_name());
    ngraph::copy_runtime_info(operation, newOperation);

    auto op = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(newOperation);
    if (op != nullptr) {
        if (updatePrecision) {
            op->set_overridden_output_type(newOperation->get_input_element_type(0));
        } else if (dequantization.multiply) {
            op->set_overridden_output_type(dequantization.multiply->get_input_element_type(1));
        } else if (dequantization.subtract) {
            op->set_overridden_output_type(dequantization.subtract->get_input_element_type(1));
        }
        std::dynamic_pointer_cast<ngraph::Node>(newOperation)->validate_and_infer_types();
    }

    const element::Type deqPrecision = dequantization.multiply->get_input_node_shared_ptr(1)->get_output_element_type(0);
    const bool shouldConvert = (newOperation->get_output_element_type(0) != deqPrecision);

    auto parent = newOperation;
    if (shouldConvert) {
        const auto convertOutputPrecision = dequantization.convert != nullptr ?
            dequantization.convert->get_output_element_type(0) :
            deqPrecision;
        parent = std::make_shared<DequantizationConvert>(parent, convertOutputPrecision);
        ngraph::copy_runtime_info({ newOperation, parent }, parent);
    }

    if (moveSubtract && (dequantization.subtract != nullptr)) {
        if (dequantization.subtractConvert == nullptr) {
            const element::Type parentPrecision = parent->get_output_element_type(0);
            if (parentPrecision.bitwidth() < dequantization.subtractConstant->output(0).get_element_type().bitwidth()) {
                THROW_IE_LPT_EXCEPTION(*parent) <<
                    "unexpected precisions: on data " << parent->get_friendly_name() << ":" << parentPrecision <<
                    ", subtract dequantization constant " << dequantization.subtractConstant->get_friendly_name() << ":" <<
                    dequantization.subtractConstant->output(0).get_element_type();
            }

            parent = std::make_shared<DequantizationSubtract>(
                parent,
                dequantization.subtractConstant->output(0).get_element_type() == parentPrecision ?
                    dequantization.subtractConstant :
                    foldConvert(dequantization.subtractConstant, parentPrecision));
            ngraph::copy_runtime_info({ newOperation, parent }, parent);
        } else {
            parent = std::make_shared<DequantizationSubtract>(parent, dequantization.subtractConvert);
            ngraph::copy_runtime_info({ newOperation, parent }, parent);
        }
    }

    if (dequantization.multiply != nullptr) {
        auto multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);
        const element::Type parentPrecision = parent->get_output_element_type(0);
        if (parentPrecision.bitwidth() < multiplyConstant->output(0).get_element_type().bitwidth()) {
            THROW_IE_LPT_EXCEPTION(*parent) <<
                "unexpected precisions: on data " << parent->get_friendly_name() << ":" << parentPrecision <<
                ", multiply dequantization constant " << multiplyConstant->get_friendly_name() << ":" << multiplyConstant->output(0).get_element_type();
        }

        parent = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            DequantizationMultiply(parent,
                multiplyConstant->output(0).get_element_type() == parentPrecision ?
                    multiplyConstant :
                    foldConvert(multiplyConstant->output(0), parentPrecision)),
            dequantization.multiply->get_output_element_type(0));
        ngraph::copy_runtime_info({ newOperation, parent }, parent);
    }
    replace_node(operation, parent);

    if ((!moveSubtract) && (dequantization.convert != nullptr) && (dequantization.subtract != nullptr)) {
        NetworkHelper::cleanRunTimeInfo(dequantization.subtract);
        // issue #43088
        // NetworkHelper::optimizeElementwise(dequantization.subtract);
    }

    return InsertDequantizationResult(newOperation, parent);
}

bool NetworkHelper::checkConstantValuePrecision(const element::Type expectedPrecision, const std::shared_ptr<Node>& constant) {
    if (expectedPrecision.is_signed()) {
        return true;
    }

    std::shared_ptr<opset1::Constant> constantOp = as_type_ptr<opset1::Constant>(constant);
    if (constantOp == nullptr) {
        return false;
    }

    const auto values = constantOp->cast_vector<float>();
    const bool convertCanBeRemoved =
        (expectedPrecision.is_signed() || (std::all_of(values.begin(), values.end(), [](const float value) { return value >= 0.f; })));
    return convertCanBeRemoved;
}

size_t NetworkHelper::getChildInputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child) {
    for (size_t i = 0; i < child->get_input_size(); ++i) {
        if (parent.get() == child->get_input_node_ptr(i)) {
            return i;
        }
    }
    THROW_IE_LPT_EXCEPTION(*child) << "child input index between " <<
        parent->get_friendly_name() << " and " << child->get_friendly_name() << " was not found";
}

size_t NetworkHelper::getParentOutputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child) {
    for (size_t i = 0; i < parent->get_output_size(); ++i) {
        const auto& targetInputs = parent->output(i).get_target_inputs();
        for (const auto& targetInput : targetInputs) {
            if (targetInput.get_node() == child.get()) {
                return i;
            }
        }
    }
    THROW_IE_LPT_EXCEPTION(*child) << "parent output index between " <<
        parent->get_friendly_name() << " and " << child->get_friendly_name() << " was not found";
}

std::shared_ptr<Node> NetworkHelper::toScalarIfPossible(std::shared_ptr<Node> node) {
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(node);
    if (constant == nullptr) {
        return node;
    }

    if (!NetworkHelper::isScalarLike(constant)) {
        return node;
    }

    return NetworkHelper::toScalar(constant);
}

std::shared_ptr<Node> foldConvert(const Output<Node>& node, const element::Type targetPrecision) {
    if (is_type<opset1::Constant>(node.get_node_shared_ptr()) && (node.get_element_type() == targetPrecision)) {
        return node.get_node_shared_ptr();
    }

    return fold<opset1::Convert>(node, targetPrecision);
}

bool NetworkHelper::checkZeroPoint(const std::shared_ptr<Node>& node, const DataPrecision& dataPrecision) {
    if (!node) {
        return true;
    }

    float min, max;
    if (is_type<opset1::Subtract>(node)) {
        const auto parent = node->get_input_node_shared_ptr(0);
        const auto intNode = is_type<opset1::Convert>(parent) ? parent : node;
        const auto type = intNode->get_input_element_type(0);
        if (type == element::u8 || type == element::i8) {
            min = DataPrecision::getMinValue(type, 256) - 0.5f;
            max = DataPrecision::getMaxValue(type, 256) + 0.5f;
        } else {
            return type == element::f32 || type == element::f16;
        }
        auto subtract1input = node->get_input_node_shared_ptr(1);
        if (is_type<opset1::Convert>(subtract1input)) {
            return true;
        }
        auto subtractConst = as_type_ptr<opset1::Constant>(subtract1input);
        if (!subtractConst) {
            subtractConst = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(1)->get_input_node_shared_ptr(0));
            if (subtractConst == nullptr) {
                return false;
            }
        }
        const auto subtractValues = subtractConst->cast_vector<float>();
        if (std::any_of(subtractValues.begin(), subtractValues.end(), [min, max] (const float& val) {
                return (val < min) || (val > max); })) {
            return false;
        }
    } else if (is_type<opset1::FakeQuantize>(node)) {
        if (!dataPrecision.hasZeroPoint) {
            return true;
        }
        min = dataPrecision.min - 0.5f;
        max = dataPrecision.max + 0.5f;
        const auto quantizationDetails = QuantizationDetails::getDetails(as_type_ptr<opset1::FakeQuantize>(node));
        for (size_t i = 0; i < quantizationDetails.outputIntervalsCount; ++i) {
            float shift;
            if (quantizationDetails.outputHighValues[i] != quantizationDetails.outputLowValues[i]) {
                shift = (dataPrecision.min * quantizationDetails.outputHighValues[i] -
                         dataPrecision.max * quantizationDetails.outputLowValues[i]) /
                        (quantizationDetails.outputHighValues[i] - quantizationDetails.outputLowValues[i]);
            } else {
                shift = 0.f;
            }
            if (shift < min || shift > max) {
                return false;
            }
        }
    }
    return true;
}

std::vector<element::Type> NetworkHelper::precisionIntersection(
        const std::vector<element::Type>& v1,
        const std::vector<element::Type>& v2) noexcept {
    std::vector<element::Type> v3;

    auto v1Copy = v1;
    auto v2Copy = v2;

    std::sort(v1Copy.begin(), v1Copy.end());
    std::sort(v2Copy.begin(), v2Copy.end());

    std::set_intersection(v1Copy.begin(), v1Copy.end(),
                          v2Copy.begin(), v2Copy.end(),
                          std::back_inserter(v3));
    return v3;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
