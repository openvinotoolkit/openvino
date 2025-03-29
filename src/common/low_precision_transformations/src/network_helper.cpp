// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset6.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
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
        return ov::is_type<ov::opset1::Parameter>(node) ||
            ov::is_type<ov::opset1::Convolution>(node) ||
            ov::is_type<ov::opset1::GroupConvolution>(node) ||
            ov::is_type<ov::opset1::MatMul>(node) ||
            ov::is_type<ov::opset1::ConvolutionBackpropData>(node) ||
            ov::is_type<ov::opset3::ReadValue>(node) ||
            ov::is_type<ov::opset6::ReadValue>(node);
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

std::shared_ptr<ov::opset1::Constant> NetworkHelper::foldDequantizationConstant(
    const std::shared_ptr<ov::opset1::Constant>& foldingConstant,
    const std::shared_ptr<Node>& operation,
    const size_t outIdx) {
    OutputVector inputs = operation->input_values();
    OutputVector outputs(operation->get_output_size());

    if (shape_size(foldingConstant->get_shape()) == 1ul) {
        return toScalar(foldingConstant);
    } else {
        inputs[0] = foldingConstant;
        const auto op = operation->clone_with_new_inputs(inputs);

        if (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(op)) {
            setOutDataPrecisionForTypeRelaxed(op, inputs[0].get_element_type());
        }

        // constant folding of constant
        op->constant_fold(outputs, inputs);

        const auto result = ov::as_type_ptr<ov::opset1::Constant>(outputs[outIdx].get_node_shared_ptr());
        if (result == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "result of constant folding is not constant";
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
    if (ov::is_type<ov::opset1::Convolution>(layer)) {
        return 1;
    } else if (ov::is_type<ov::opset1::GroupConvolution>(layer)) {
        return layer->get_input_partial_shape(1)[0].get_length();    // input weights for opset1::GC is in format GOI..., see the specification
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid layer type of " << layer->get_friendly_name() << "; expected Convolution or GroupConvolution";
    }
}

void NetworkHelper::removeLayer(std::shared_ptr<Node> layer) {
    ov::replace_output_update_name(layer->output(0), layer->input_value(0));
}

std::shared_ptr<Node> NetworkHelper::swapMultiplyAndAdd(std::shared_ptr<ov::opset1::Add> addAfterMultiply, const int multiplyBranch) {
    // Multiply --> Add(addAfterMultiply)  ==>  Add(new) --> Multiply(new)
    // That means x*a + b ==> (x + b/a)*a; tries to fold b/a
    const auto multiply = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch);

    const auto multiplyParent1 = multiply->get_input_node_shared_ptr(0);
    const auto multiplyParent2 = multiply->get_input_node_shared_ptr(1);

    auto multiplyInput = ov::as_type_ptr<ov::opset1::Multiply>(multiplyParent1);
    auto multiplyConst = ov::as_type_ptr<ov::opset1::Constant>(multiplyParent2);
    int multiplyInputBranch = 0;

    if (multiplyConst == nullptr) {
        multiplyInput = ov::as_type_ptr<ov::opset1::Multiply>(multiplyParent2);
        multiplyConst = ov::as_type_ptr<ov::opset1::Constant>(multiplyParent1);
        multiplyInputBranch = 1;
    }

    if (multiplyConst == nullptr)
        return addAfterMultiply;

    auto a = as_type_ptr<ov::opset1::Constant>(multiply->get_input_node_shared_ptr(multiplyInputBranch == 0 ? 1 : 0));
    auto b = as_type_ptr<ov::opset1::Constant>(addAfterMultiply->get_input_node_shared_ptr(multiplyBranch == 0 ? 1 : 0));
    std::shared_ptr<ov::opset1::Constant> bDivA;

    const auto aPShape = a->get_output_partial_shape(0);
    assert(aPShape.is_static());
    const auto aShape = aPShape.to_shape();

    const auto bPShape = b->get_output_partial_shape(0);
    assert(bPShape.is_static());
    const auto bShape = bPShape.to_shape();

    if ((shape_size(bShape) == 1) || (shape_size(aShape) == 1) || (shape_size(bShape) == shape_size(aShape))) {
        // safely division to avoid NaN
        const std::vector<float> bValues = b->cast_vector<float>();
        const std::vector<float> aValues = a->cast_vector<float>();
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
        bDivA = std::make_shared<ov::opset1::Constant>(
                aPrecision,
                aBroadcasted ? bShape : aShape,
                bDivAValues);
    } else {
        b = as_type_ptr<ov::opset1::Constant>(foldConvert(b->output(0), element::f32));
        a = as_type_ptr<ov::opset1::Constant>(foldConvert(a->output(0), element::f32));
        bDivA = as_type_ptr<ov::opset1::Constant>(fold<ov::opset1::Divide>(b->output(0), a->output(0)));
        // TODO: issue #49868
        bDivA = as_type_ptr<ov::opset1::Constant>(foldConvert(bDivA->output(0), a->get_element_type()));
    }

    const auto& add_input = multiply->input_value(multiplyInputBranch);
    // Note: precision is copied to a separate variable intentionally,
    // since TemporaryReplaceOutputType replaces add_input's precision, whereas we need to set the original precision on newAdd's output
    const auto add_output_precision = add_input.get_element_type();
    std::shared_ptr<ov::opset1::Add> newAdd = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Add>>(
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{ add_output_precision },
        ov::op::TemporaryReplaceOutputType(add_input, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(bDivA, element::f32).get());
    copyInfo(addAfterMultiply, newAdd);

    auto newMultiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{ multiply->get_output_element_type(0) },
            ov::op::TemporaryReplaceOutputType(newAdd->output(0), element::f32).get(),
            ov::op::TemporaryReplaceOutputType(a->output(0), element::f32).get());
    copyInfo({ multiply, newMultiply }, newMultiply);

    replace_node(addAfterMultiply, newMultiply);
    return newMultiply;
}

void NetworkHelper::copyInfo(
    const std::vector<std::shared_ptr<Node>>& sources,
    const std::vector<std::shared_ptr<Node>>& targets,
    bool overrideName) {
    ov::copy_runtime_info(sources, targets);

    for (const auto& target : targets) {
        const std::string friendlyName = sources[0]->get_friendly_name();
        if (!friendlyName.empty() && overrideName) {
            target->set_friendly_name(friendlyName);
        }

        {
            // TODO: has to be implemented in ov::copy_runtime_info

            for (auto& source : sources) {
                if (target->get_type_info() != source->get_type_info()) {
                    continue;
                }

                assert(source->get_input_size() == target->get_input_size());
                for (size_t i = 0; i < target->get_input_size(); ++i) {
                    auto sourceInput = source->input(i);
                    const auto& sourceRt = sourceInput.get_rt_info();
                    auto targetInput = target->input(i);
                    auto& targetRt = targetInput.get_rt_info();
                    for (const auto& it : sourceRt) {
                        targetRt[it.first] = it.second;
                    }
                }

                assert(source->get_output_size() == target->get_output_size());
                for (size_t i = 0; i < target->get_output_size(); ++i) {
                    auto sourceOutput = source->output(i);
                    const auto& sourceRt = sourceOutput.get_rt_info();
                    auto targetOutput = target->output(i);
                    auto& targetRt = targetOutput.get_rt_info();
                    for (const auto& it : sourceRt) {
                        targetRt[it.first] = it.second;
                    }
                }
            }
        }
    }
}

void NetworkHelper::copyInfo(const std::vector<std::shared_ptr<Node>>& sources, const std::shared_ptr<Node>& target, bool overrideName) {
    copyInfo(sources, std::vector<std::shared_ptr<Node>>{ target }, overrideName);
}

void NetworkHelper::copyInfo(const std::shared_ptr<Node>& source, const std::shared_ptr<Node>& target, bool overrideName) {
    copyInfo(std::vector<std::shared_ptr<Node>>{ source }, std::vector<std::shared_ptr<Node>>{ target }, overrideName);
}

bool NetworkHelper::isScalarLike(std::shared_ptr<ov::opset1::Constant> constant) {
    // ticket #48857
    // return constant->get_all_data_elements_bitwise_identical();

    const auto shape = constant->output(0).get_shape();
    if (shape_size(shape) == 1ul) {
        return true;
    }


    const auto values = constant->cast_vector<float>();
    if (values.empty()) {
        return true;
    }

    return !std::any_of(values.begin(), values.end(), [&](float value) { return values[0] != value; });
}

bool NetworkHelper::isZero(std::shared_ptr<ov::opset1::Constant> constant) {
    static const float minQuantizationShift = 1e-32f;

    auto values = constant->cast_vector<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        if (fabs(values[i]) > minQuantizationShift) {
            return false;
        }
    }

    return true;
}

std::shared_ptr<ov::opset1::Constant> NetworkHelper::toScalar(std::shared_ptr<ov::opset1::Constant> constant) {
    assert(isScalarLike(constant));
    return std::make_shared<ov::opset1::Constant>(constant->get_element_type(), Shape{}, constant->get_data_ptr());
}

std::shared_ptr<Node> NetworkHelper::getConstantInput(const std::shared_ptr<const Node>& node, const bool convertIsExpected) {
    std::shared_ptr<Node> parent = ov::as_type_ptr<ov::opset1::Constant>(node->input_value(0).get_node_shared_ptr());
    if (parent != nullptr) {
        return parent;
    }

    parent = ov::as_type_ptr<ov::opset1::Constant>(node->input_value(1).get_node_shared_ptr());
    if (parent != nullptr) {
        return parent;
    }

    if (convertIsExpected) {
        auto getConstantBeforeConvert = [](const std::shared_ptr<Node>& node) -> std::shared_ptr<Node> {
            std::shared_ptr<Node> parent = ov::as_type_ptr<ov::opset1::Convert>(node);
            if (parent != nullptr) {
                parent = ov::as_type_ptr<ov::opset1::Constant>(parent->input_value(0).get_node_shared_ptr());
                if (parent != nullptr) {
                    return parent;
                }
            }
            return nullptr;
        };

        parent = getConstantBeforeConvert(node->input_value(0).get_node_shared_ptr());
        if (parent != nullptr) {
            return parent;
        }

        parent = getConstantBeforeConvert(node->input_value(1).get_node_shared_ptr());
        if (parent != nullptr) {
            return parent;
        }
    }

    return nullptr;
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
    return std::move(updatedReshapeValues);
}

std::shared_ptr<ov::opset1::Multiply> NetworkHelper::optimizeMultipliesAfter(std::shared_ptr<Node> node) {
    const auto multiply = ov::as_type_ptr<ov::opset1::Multiply>(std::move(node));
    if (!multiply) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected operation type in the optimizeMultipliesAfter method";
    }

    if (multiply->output(0).get_target_inputs().size() == 1) {
        auto constant1 = getConstantInput(multiply);
        if (!constant1 || constant1->output(0).get_target_inputs().size() != 1) {
            return multiply;
        }

        auto nextMultiplyInput = *multiply->output(0).get_target_inputs().begin();
        auto nextMultiply = ov::as_type_ptr<ov::op::TypeRelaxed<ov::opset1::Multiply>>(nextMultiplyInput.get_node()->shared_from_this());
        if (nextMultiply) {
            auto constant2 = getConstantInput(nextMultiply);
            if (!constant2 || constant2->output(0).get_target_inputs().size() != 1) {
                return multiply;
            }

            auto newInput = multiply->input_value(1 - constant1->output(0).get_target_inputs().begin()->get_index());
            auto multiplyResult = fold<ov::opset1::Multiply>(constant1->output(0), constant2->output(0));
            {
                // optimize constant shape: used in rfcn-resnet101-coco
                const auto multiplyResultConstant = ov::as_type_ptr<ov::opset1::Constant>(multiplyResult);
                if ((multiplyResultConstant != nullptr) && NetworkHelper::isScalarLike(multiplyResultConstant)) {
                    multiplyResult = NetworkHelper::toScalar(multiplyResultConstant);
                }
            }
            auto inputPrecision0 = nextMultiply->get_origin_input_type(0);
            auto inputPrecision1 = nextMultiply->get_origin_input_type(1);
            auto outputPrecision = nextMultiply->get_overridden_output_type(0);
            auto newMultiply =
                    std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                            std::vector<element::Type>{ inputPrecision0, inputPrecision1 },
                            std::vector<element::Type>{ outputPrecision },
                            ov::op::TemporaryReplaceOutputType(newInput, inputPrecision0).get(),
                            ov::op::TemporaryReplaceOutputType(multiplyResult, inputPrecision1).get());
            copy_runtime_info(multiply, newMultiply);
            replace_node(nextMultiply, newMultiply);
            return newMultiply;
        }
    }

    return nullptr;
}

std::shared_ptr<ov::opset1::Constant> NetworkHelper::round(std::shared_ptr<Node> node, element::Type target_type) {
    const auto constant = ov::as_type_ptr<ov::opset1::Constant>(node);
    assert(constant);

    const auto castedConstant = ov::as_type_ptr<ov::opset1::Constant>(fold<ov::op::v0::Convert>(
        fold<ov::op::v5::Round>(constant->output(0), ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO),
        target_type));

    return castedConstant;
}

std::shared_ptr<Node> NetworkHelper::fold_fake_quantize(const std::shared_ptr<ov::opset1::FakeQuantize>& fq) {
    return foldFakeQuantize(fq, false, false);
}

std::shared_ptr<Node> NetworkHelper::fold_fake_quantize(const std::shared_ptr<ov::opset1::FakeQuantize>& fq,
                                                        const bool roundValues) {
    return foldFakeQuantize(fq, roundValues, true);
}

FakeQuantizeDequantization NetworkHelper::foldDequantization(const std::shared_ptr<Node>& node,
    const size_t branchIndex,
    const std::vector<ov::element::Type>& defaultPrecisions,
    const bool inPlace) {
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, branchIndex, inPlace);
    if (dequantization.empty() || (dequantization.multiply == nullptr)) {
        return dequantization;
    }

    if (dequantization.convert != nullptr) {
        const std::shared_ptr<Node> result = foldConvert(dequantization.data, dequantization.convert->get_element_type());
        if (ov::is_type<ov::opset1::Constant>(result)) {
            if (inPlace) {
                copyInfo(dequantization.convert, result);
            }
            replace_node(dequantization.convert, result);
            dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, branchIndex, inPlace);
        }
    }

    if (dequantization.subtract != nullptr) {
        if (dequantization.subtract->get_input_element_type(0) != dequantization.subtract->get_input_element_type(1)) {
            return dequantization;
        }

        if (dequantization.subtractConvert != nullptr) {
            const auto convertionResult = foldConvert(
                dequantization.subtractConstant->output(0),
                dequantization.subtractConvert->get_element_type());
            if (ov::is_type<ov::opset1::Constant>(convertionResult)) {
                replace_node(dequantization.subtractConvert, convertionResult);
                dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, branchIndex, inPlace);
            }
        }

        const std::shared_ptr<Node> result = fold<ov::opset1::Subtract>(
            dequantization.subtract->input_value(0),
            dequantization.subtract->input_value(1));
        if (ov::is_type<ov::opset1::Constant>(result)) {
            if (inPlace) {
                copyInfo(dequantization.subtract, result);
            }
            replace_node(dequantization.subtract, result);
            dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, branchIndex, inPlace);
        } else {
            return dequantization;
        }
    }

    if (dequantization.multiply != nullptr) {
        if (dequantization.multiply->get_input_element_type(0) != dequantization.multiply->get_input_element_type(1)) {
            return dequantization;
        }

        std::shared_ptr<Node> result = fold<ov::opset1::Multiply>(
                dequantization.multiply->input_value(0),
                dequantization.multiply->input_value(1));
        if (!ov::is_type<ov::opset1::Constant>(result)) {
            return dequantization;
        }
        if (dequantization.multiply->get_output_element_type(0) != result->get_element_type()) {
            result = foldConvert(result->output(0), dequantization.multiply->get_output_element_type(0));
        }
        if (inPlace) {
            copyInfo(dequantization.multiply, result);
        }
        replace_node(dequantization.multiply, result);
        dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, branchIndex, inPlace);
    }


    return dequantization;
}

std::shared_ptr<ov::Node> NetworkHelper::separateInStandaloneBranch(std::shared_ptr<ov::Node> node,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    auto inputs = node->input_values();
    auto separate_branch = [&](size_t input_idx) {
        const auto dequantization = NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(node, defaultPrecisions, input_idx));
        if (dequantization.empty() || !dequantization.isShared())
            return false;

        Output<Node> parent = dequantization.data;
        if (dequantization.convert != nullptr) {
            auto convert = dequantization.convert->clone_with_new_inputs({ parent });
            convert->set_friendly_name("");
            copy_runtime_info(parent.get_node_shared_ptr(), convert);
            parent = convert->output(0);
        }

        if (dequantization.subtract != nullptr) {
            const auto parentOnWeights = dequantization.subtract->get_input_node_shared_ptr(1);
            const std::vector<Input<Node>> inputs = parentOnWeights->inputs();
            OutputVector outputs;
            outputs.reserve(inputs.size());
            for (const auto& input : inputs) {
                outputs.push_back(input.get_source_output());
            }

            auto subtract = dequantization.subtract->clone_with_new_inputs({parent, parentOnWeights->clone_with_new_inputs(outputs)->output(0) });
            subtract->set_friendly_name("");
            copy_runtime_info(parent.get_node_shared_ptr(), subtract);
            parent = subtract->output(0);
        }

        if (dequantization.multiply != nullptr) {
            auto multiply = dequantization.multiply->clone_with_new_inputs({
                parent,
                dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({})->output(0) });
            multiply->set_friendly_name("");
            copy_runtime_info(parent.get_node_shared_ptr(), multiply);
            parent = multiply->output(0);
        }

        OPENVINO_ASSERT(dequantization.multiply != nullptr || dequantization.subtract != nullptr, "incorrect dequantization ops configuration");
        const auto originalParent = dequantization.multiply ?
            dequantization.multiply->shared_from_this() :
            dequantization.subtract->shared_from_this();

        const size_t inputIndex = NetworkHelper::getChildInputIndex(originalParent, node);
        inputs[inputIndex] = parent;
        return true;
    };

    bool branch_separation_happened = false;
    for (size_t i = 0; i < node->get_input_size(); ++i)
        branch_separation_happened |= separate_branch(i);

    if (!branch_separation_happened)
        return node;

    const auto newNode = node->clone_with_new_inputs(inputs);
    copy_runtime_info(node, newNode);
    replace_node(node, newNode);
    newNode->set_friendly_name(node->get_friendly_name());
    return newNode;
}

std::shared_ptr<ov::opset1::FakeQuantize> NetworkHelper::fuseConvert(const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize) {
    const Output<Node> output = fakeQuantize->output(0);
    const auto targetInputs = output.get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return fakeQuantize;
    }

    Node* node = targetInputs.begin()->get_node();
    if (!ov::is_type<ov::opset1::Convert>(node) ||
        // TODO: LPT: avoid precision hardcode: to separate method: isSupportedPrecision
        ((node->get_output_element_type(0) != element::u8) && (node->get_output_element_type(0) != element::i8))) {
        return fakeQuantize;
    }


    std::shared_ptr<ov::opset1::FakeQuantize> newFakeQuantize = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
        std::vector<ov::element::Type>{ element::f32, element::f32, element::f32, element::f32, element::f32 },
        std::vector<ov::element::Type>{},
        ov::op::TemporaryReplaceOutputType(fakeQuantize->input_value(0), element::f32).get(),
        ov::op::TemporaryReplaceOutputType(fakeQuantize->input_value(1), element::f32).get(),
        ov::op::TemporaryReplaceOutputType(fakeQuantize->input_value(2), element::f32).get(),
        ov::op::TemporaryReplaceOutputType(fakeQuantize->input_value(3), element::f32).get(),
        ov::op::TemporaryReplaceOutputType(fakeQuantize->input_value(4), element::f32).get(),
        fakeQuantize->get_levels());
    NetworkHelper::setOutDataPrecisionForTypeRelaxed(newFakeQuantize, node->get_output_element_type(0));
    newFakeQuantize->set_friendly_name(node->get_friendly_name());
    replace_node(node->shared_from_this(), newFakeQuantize);
    bool overrideName = false;
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize, overrideName);

    return newFakeQuantize;
}

bool NetworkHelper::isPrecisionPreserved(const std::shared_ptr<ov::Node>& node) {
    auto& rt = node->get_rt_info();
    auto it = rt.find(PrecisionPreservedAttribute::get_type_info_static());
    if (it == rt.end()) {
        return false;
    }
    auto attribute = it->second;
    return attribute.as<PrecisionPreservedAttribute>().value();
}

size_t NetworkHelper::calculateLevels(
    const float dataPrecisionMin,
    const float dataPrecisionMax,
    const float combinedIntervalLow,
    const float combinedIntervalHigh,
    const float minIntervalLow,
    const float minIntervalHigh,
    float& dequantizationMul,
    float& dequantizationSub,
    float& updatedOutputLowValue,
    float& updatedOutputHighValue) {
    if (combinedIntervalHigh == combinedIntervalLow) {
        // degenerate case: quantization to a point
        dequantizationMul = 1.f;
        dequantizationSub = 0.f;
        updatedOutputLowValue = combinedIntervalLow;
        updatedOutputHighValue = combinedIntervalHigh;
        return 1ull;
    }

    const float maxOutputInterval = combinedIntervalHigh - combinedIntervalLow;
    // FQ -> SUB_quantization -> MUL_quantization -[INT8]-> SUB_dequantization -> MUL_dequantization ->
    const float quantizationMul = (dataPrecisionMax - dataPrecisionMin) / maxOutputInterval;
    dequantizationMul = maxOutputInterval / (dataPrecisionMax - dataPrecisionMin);

    // FQ outputLowValue = dataPrecision.min * dequantizationMul - quantizationSub
    const float quantizationSub = combinedIntervalLow - dataPrecisionMin * dequantizationMul;
    dequantizationSub = std::round(-quantizationSub * quantizationMul);

    updatedOutputLowValue = (minIntervalLow - quantizationSub) * quantizationMul;
    updatedOutputHighValue = (minIntervalHigh - quantizationSub) * quantizationMul;

    const size_t levels = static_cast<size_t>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
    return levels;
}

std::shared_ptr<Node> NetworkHelper::foldFakeQuantize(
    const std::shared_ptr<ov::opset1::FakeQuantize>& fq,
    const bool roundValuesArg,
    const bool roundValuesWasSet) {
    // Corner case
    //    y = FakeQuantize(x, inputLow, inputHigh, outputLow, outputHigh)
    // given:
    //    outputLow is const
    //    outputHigh is const
    //    outputLow == outputHigh
    // => y = Broadcast(Convert(outputLow, typeof(y))), ShapeOf(x))
    if (ov::is_type<ov::opset1::Constant>(fq->get_input_node_shared_ptr(3)) &&
        ov::is_type<ov::opset1::Constant>(fq->get_input_node_shared_ptr(4))) {
        const auto outputLowValues =
            ov::as_type_ptr<ov::opset1::Constant>(fq->get_input_node_shared_ptr(3))->cast_vector<float>();
        const auto outputHighValues =
            ov::as_type_ptr<ov::opset1::Constant>(fq->get_input_node_shared_ptr(4))->cast_vector<float>();

        if (outputLowValues == outputHighValues) {
            const auto data_shape_node = fold<ov::opset1::ShapeOf>(fq->input_value(0));
            const auto cvt_output_low = foldConvert(fq->input_value(3), fq->get_output_element_type(0));
            return fold<ov::opset1::Broadcast>(cvt_output_low, data_shape_node);
        }
    }

    if (ov::is_type<ov::opset1::Constant>(fq->get_input_node_shared_ptr(0))) {
        std::shared_ptr<Node> subgraph = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(*fq, element::f32);
        const auto& original_et = fq->get_output_element_type(0);
        const bool roundValues = roundValuesWasSet ? roundValuesArg : original_et.is_integral();
        if (roundValues) {
            subgraph = std::make_shared<ov::opset6::Round>(subgraph, ov::opset6::Round::RoundMode::HALF_TO_EVEN);
        }

        const auto result = ov::util::get_constant_from_source(subgraph);
        if (result != nullptr) {
            return foldConvert(result, original_et);
        }
    }

    return fq;
}

std::shared_ptr<ov::opset1::FakeQuantize> NetworkHelper::composeFakeQuantize(const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    std::shared_ptr<Node> parent = fakeQuantize;
    auto targetInputs = parent->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return nullptr;
    }
    if (ov::is_type<ov::opset1::Convert>(targetInputs.begin()->get_node())) {
        parent = targetInputs.begin()->get_node()->shared_from_this();
    }

    targetInputs = parent->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return nullptr;
    }
    if (ov::is_type<ov::opset1::Subtract>(targetInputs.begin()->get_node())) {
        parent = targetInputs.begin()->get_node()->shared_from_this();
    }

    targetInputs = parent->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return nullptr;
    }
    if (ov::is_type<ov::opset1::Multiply>(targetInputs.begin()->get_node())) {
        parent = targetInputs.begin()->get_node()->shared_from_this();
    }

    const std::shared_ptr<Node> prev = parent;
    parent = parent->output(0).get_target_inputs().begin()->get_node()->shared_from_this();

    const size_t index = NetworkHelper::getChildInputIndex(prev, parent);
    const FakeQuantizeDequantization dequantization = getDequantization(parent, defaultPrecisions, index);
    if (dequantization.empty()) {
        return nullptr;
    }

    std::shared_ptr<ov::opset1::FakeQuantize> newFakeQuantize = fakeQuantize;

    if (dequantization.convert != nullptr) {
        const std::shared_ptr<ov::opset1::FakeQuantize> replacement = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
            newFakeQuantize->input_value(0),
            newFakeQuantize->input_value(1),
            newFakeQuantize->input_value(2),
            newFakeQuantize->input_value(3),
            newFakeQuantize->input_value(4),
            newFakeQuantize->get_levels(),
            newFakeQuantize->get_auto_broadcast());
        replace_node(dequantization.convert, replacement);
        //replacement->set_friendly_name(newFakeQuantize->get_friendly_name());
        copyInfo({ fakeQuantize, dequantization.convert }, replacement);
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, dequantization.convert->output(0).get_element_type());
        newFakeQuantize = replacement;
    }

    if (dequantization.subtract != nullptr) {
        const auto subtractValue = (dequantization.subtractConvert == nullptr) ?
            dequantization.subtractConstant :
            foldConvert(dequantization.subtractConstant->output(0), dequantization.subtractConvert->get_destination_type());

        const std::shared_ptr<ov::opset1::FakeQuantize> replacement = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
            newFakeQuantize->input_value(0),
            newFakeQuantize->input_value(1),
            newFakeQuantize->input_value(2),
            fold<ov::opset1::Subtract>(newFakeQuantize->input_value(3), subtractValue),
            fold<ov::opset1::Subtract>(newFakeQuantize->input_value(4), subtractValue),
            newFakeQuantize->get_levels(),
            newFakeQuantize->get_auto_broadcast());
        replace_node(dequantization.subtract, replacement);
        //replacement->set_friendly_name(newFakeQuantize->get_friendly_name());
        copyInfo({ newFakeQuantize, dequantization.subtract }, replacement);
        newFakeQuantize = replacement;
    }

    if (dequantization.multiply != nullptr) {
        // multiply different precision constants (value1 & value2) and convert result to first argument precision (value1)
        auto multiply = [](const Output<Node>& value1, const Output<Node>& value2) {
            const ov::element::Type precision1 = value1.get_element_type();
            const ov::element::Type precision2 = value2.get_element_type();
            // 1) precision1 & precision2 are not equal but similar
            // 2) precision2 >= precision1
            assert((precision2.is_real() == precision1.is_real()) && (precision2.bitwidth() >= precision1.bitwidth()));

            auto output = fold<ov::opset1::Multiply>(
                precision2 != precision1 ? foldConvert(value1, precision2) : value1,
                value2);

            if (output->output(0).get_element_type() != precision1) {
                output = foldConvert(output->output(0), precision1);
            }

            return output;
        };

        const std::shared_ptr<ov::opset1::FakeQuantize> replacement = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
            newFakeQuantize->input_value(0ul),
            newFakeQuantize->input_value(1ul),
            newFakeQuantize->input_value(2ul),
            multiply(newFakeQuantize->input_value(3ul), dequantization.multiplyConstant),
            multiply(newFakeQuantize->input_value(4ul), dequantization.multiplyConstant),
            newFakeQuantize->get_levels(),
            newFakeQuantize->get_auto_broadcast());

        replace_node(dequantization.multiply, replacement);
        //replacement->set_friendly_name(newFakeQuantize->get_friendly_name());
        copyInfo({ newFakeQuantize, dequantization.multiply }, replacement);
        newFakeQuantize = replacement;
    }

    return newFakeQuantize;
}

// Decompose FakeQuantize to FakeQuantize with output integer limits (quantize), dequatized MultiplyAdd
// To align types the resulting sequence is FakeQuantize -> Convert -> Convert -> MultiplyAdd
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> NetworkHelper::decomposeFakeQuantize(
    std::shared_ptr<ov::opset1::FakeQuantize> fq,
    const element::Type precision,
    const float min,
    const float max,
    const bool hasZeroPoint,
    const bool updatePrecision,
    const element::Type deqPrecision,
    const size_t outChannelsShapeIndex) {
    const auto outputLow = fq->input_value(3);
    const auto outputHigh = fq->input_value(4);

    std::vector<float> outputLowValues = ov::as_type_ptr<ov::opset1::Constant>(outputLow.get_node_shared_ptr())->cast_vector<float>();
    std::vector<float> outputHighValues = ov::as_type_ptr<ov::opset1::Constant>(outputHigh.get_node_shared_ptr())->cast_vector<float>();
    size_t outputSize = outputLowValues.size();
    std::vector<float> minValues(outputSize, min);
    std::vector<float> maxValues(outputSize, max);
    std::vector<float> shifts(outputSize, 0.f);
    std::vector<float> scales(outputSize);

    for (size_t i = 0; i < outputSize; ++i) {
        if (outputHighValues[i] != outputLowValues[i]) {
            // compute dequantizations (in double for INT32)
            if (precision == element::i32 || precision == element::u32) {
                shifts[i] = static_cast<float>(
                            (static_cast<double>(min) * outputHighValues[i] - static_cast<double>(max) * outputLowValues[i]) /
                            (static_cast<double>(outputHighValues[i]) - outputLowValues[i]));
                scales[i] = static_cast<float>(
                        (static_cast<double>(outputHighValues[i]) - outputLowValues[i]) / (static_cast<double>(max) - min));
            } else {
                shifts[i] = (min * outputHighValues[i] - max * outputLowValues[i]) /
                            (outputHighValues[i] - outputLowValues[i]);
                scales[i] = (outputHighValues[i] - outputLowValues[i]) / (max - min);
            }
            if (shifts[i] == -0.f) {
                shifts[i] = 0.f;
            }
        } else {
            scales[i] = 1.f;
            minValues[i] = outputHighValues[i];
            maxValues[i] = outputHighValues[i];
        }
    }

    if ((!updatePrecision) &&
        std::all_of(scales.begin(), scales.end(), [](const float value) { return value == 1.f; }) &&
        std::all_of(shifts.begin(), shifts.end(), [](const float value) { return value == 0.f; })) {
        return std::make_tuple(nullptr, nullptr);
    }

    std::shared_ptr<Node> shift = hasZeroPoint ?
        std::make_shared<ov::opset1::Constant>(deqPrecision, outputLow.get_shape(), shifts) :
        nullptr;
    std::shared_ptr<Node> scale = std::make_shared<ov::opset1::Constant>(element::f32, outputLow.get_shape(), scales);

    auto newMin = std::make_shared<ov::opset1::Constant>(outputLow.get_element_type(), outputLow.get_shape(), minValues);
    auto newMax = std::make_shared<ov::opset1::Constant>(outputLow.get_element_type(), outputLow.get_shape(), maxValues);

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
            scale = std::make_shared<ov::opset1::Constant>(scale->output(0).get_element_type(), scale->output(0).get_shape(), scaleValues);
        }
    }

    if ((shift != nullptr) && isZero(ov::as_type_ptr<ov::opset1::Constant>(shift))) {
        shift = nullptr;
    }

    // Build a substitution sub-graph:

    std::shared_ptr<ov::Node> newFQ = fold_fake_quantize(
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
            fq->input_value(0),
            getSingleConsumerConstant(fq->input_value(1)),
            getSingleConsumerConstant(fq->input_value(2)),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast()),
        true);
    NetworkHelper::copyInfo(fq, newFQ);

    std::shared_ptr<ov::Node> convert2;
    if (updatePrecision) {
        std::shared_ptr<Node> convert;

        if (ov::is_type<ov::opset1::FakeQuantize>(newFQ)) {
            newFQ = setOutDataPrecision(ov::as_type_ptr<ov::opset1::FakeQuantize>(newFQ), precision);
            convert = newFQ;
        } else {
            convert = foldConvert(newFQ->output(0), precision);
        }

        convert2 = std::make_shared<ov::opset1::Convert>(convert, element::f32);
        convert2->set_friendly_name(convert->get_friendly_name() + "/DequantizationConvert");
        ov::copy_runtime_info({ newFQ, convert2 }, convert2);
    } else {
        if (newFQ->get_output_element_type(0) != element::f32) {
            convert2 = std::make_shared<ov::opset1::Convert>(newFQ, element::f32);
            convert2->set_friendly_name(newFQ->get_friendly_name() + "/DequantizationConvert");
            ov::copy_runtime_info({ newFQ, convert2 }, convert2);
        }
    }

    // TODO: why type relaxed?
    const std::shared_ptr<ov::Node> sub = shift == nullptr ?
        nullptr :
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(convert2 == nullptr ? newFQ : convert2, shift);
    if (sub != nullptr) {
        sub->set_friendly_name(newFQ->get_friendly_name() + "/DequantizationSubtract");
        ov::copy_runtime_info({ newFQ, sub }, sub);
    }

    const auto dequantize =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ fq->get_output_element_type(0) },
            ov::op::TemporaryReplaceOutputType(sub == nullptr ? (convert2 == nullptr ? newFQ : convert2) : sub, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(scale, element::f32).get());
    dequantize->set_friendly_name(newFQ->get_friendly_name() + "/DequantizationMultiply");
    ov::copy_runtime_info({ newFQ, dequantize }, dequantize);

    insertDequantizationAfter(fq, dequantize, newFQ);

    return std::make_tuple(newFQ, dequantize);
}

std::shared_ptr<ov::opset1::FakeQuantize> NetworkHelper::updateFakeQuantize(
    std::shared_ptr<ov::opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max,
    const bool replace) {
    auto newInMin = getSingleConsumerConstant(fq->input_value(1));
    auto newInMax = getSingleConsumerConstant(fq->input_value(2));
    auto newOutMin = std::make_shared<ov::opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newOutMax = std::make_shared<ov::opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    std::shared_ptr<ov::opset1::FakeQuantize> newFQ = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
            fq->input_value(0),
            newInMin,
            newInMax,
            newOutMin->output(0),
            newOutMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast());

    NetworkHelper::setOutDataPrecision(newFQ, precision);
    if (replace) {
        replace_node(fq, newFQ);
    }

    newFQ->set_friendly_name(fq->get_friendly_name());
    return newFQ;
}

FakeQuantizeDequantization NetworkHelper::makeDequantization(
    const float dequantizationMul,
    const float dequantizationSub,
    const ov::element::Type originalPrecision,
    const ov::PartialShape& dataNodeOutputShape,
    element::Type precision,
    const ov::element::Type deqPrecision,
    std::shared_ptr<ov::Node> input) {
    if (input == nullptr) {
        // TODO: we create input here! we really need it here?
        input = std::make_shared<ov::opset1::Parameter>(precision, dataNodeOutputShape);
    }
    std::shared_ptr<ov::Node> parent = input;

    std::shared_ptr<ov::opset1::Convert> convert;
    if (precision == deqPrecision) {
        convert = nullptr;
    } else {
        convert = std::make_shared<ov::opset1::Convert>(
            parent,
            deqPrecision);
        parent = convert;
    }

    std::shared_ptr<ov::opset1::Subtract> subtract;
    std::shared_ptr<ov::opset1::Constant> subtractConstant;
    if (std::abs(dequantizationSub) > 1e-6) {
        subtract = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
            parent,
            std::make_shared<ov::opset1::Constant>(deqPrecision, ov::Shape({}), std::vector<float>({ dequantizationSub })));
        subtract->set_output_type(0, deqPrecision, subtract->get_output_partial_shape(0));
        parent = subtract;
    }

    // mandatory
    auto multiplyConstant = std::make_shared<ov::opset1::Constant>(deqPrecision, ov::Shape({}), std::vector<float>({ dequantizationMul }));
    auto multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        ov::opset1::Multiply(parent, multiplyConstant),
        originalPrecision);

    return FakeQuantizeDequantization(input, convert, subtract, nullptr, subtractConstant, multiply, multiplyConstant);
}

std::shared_ptr<ov::Node> NetworkHelper::makeDequantizationSubtract(
    const ov::Output<ov::Node>& parent,
    const ov::Output<ov::Node>& subtract_constant) {
    return subtract_constant.get_element_type() != parent.get_element_type()
               ? std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::opset1::Subtract>(
                     parent,
                     std::make_shared<ov::opset1::Convert>(subtract_constant, parent.get_element_type())))
               : std::make_shared<ov::opset1::Subtract>(parent, subtract_constant);
}

bool NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(const std::shared_ptr<const ov::Node>& node,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    if (!ov::is_type<ov::opset1::Subtract>(node)) {
        return false;
    }

    const auto targetInputs = node->output(0).get_target_inputs();
    if (targetInputs.size() != 1ul) {
        return false;
    }

    const auto multiply = targetInputs.begin()->get_node()->shared_from_this();
    return areQuantizeAndDequantizeSupportedForMultiply(multiply, defaultPrecisions);
}

bool NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(const std::shared_ptr<const ov::Node>& node,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    if (!ov::is_type<ov::opset1::Multiply>(node)) {
        return false;
    }

    const std::shared_ptr<ov::Node> multiply = const_cast<ov::Node*>(node.get())->shared_from_this();
    const auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(multiply, defaultPrecisions, 0, true);
    if (dequantization.empty()) {
        return false;
    }

    const auto dataNode = dequantization.data.get_node();
    if (ov::is_type<ov::opset1::Convert>(dataNode)) {
        const auto quantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(dataNode->get_input_node_shared_ptr(0));
        if (quantize == nullptr) {
            return false;
        }

        return NetworkHelper::isQuantizeSupported(quantize);
    } else if (ov::is_type<ov::opset1::Constant>(dataNode)) {
        return true;
    }

    return false;
}

bool NetworkHelper::isQuantizeSupported(const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize) {
    return QuantizationDetails::outputLayoutIsSupported(fakeQuantize) && QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels());
}

FakeQuantizeDequantization NetworkHelper::getDequantization(const std::shared_ptr<const Node>& node,
    const std::vector<ov::element::Type> defaultPrecisions,
    const size_t parentIndex,
    const bool inPlace) {
    auto getDataIndex = [](const std::shared_ptr<ov::Node>& node) {
        if (ov::is_type<ov::opset1::Constant>(node->get_input_node_ptr(1))) {
            return 0ul;
        }

        if (ov::is_type<ov::opset1::Convert>(node->get_input_node_ptr(1)) &&
            ov::is_type<ov::opset1::Constant>(node->get_input_node_ptr(1)->get_input_node_ptr(0))) {
            return 0ul;
        }

        if (ov::is_type<ov::opset1::Convert>(node->get_input_node_ptr(0)) &&
            ov::is_type<ov::opset1::Constant>(node->get_input_node_ptr(0)->get_input_node_ptr(0))) {
            return 1ul;
        }

        return 1ul;
    };

    Output<Node> dataNode;
    if (inPlace) {
        dataNode = std::const_pointer_cast<Node>(node);
    } else {
        if (parentIndex >= node->get_input_size())
            return FakeQuantizeDequantization();
        dataNode = node->input_value(parentIndex);
    }

    const std::shared_ptr<ov::opset1::Multiply> multiply = ov::as_type_ptr<ov::opset1::Multiply>(dataNode.get_node_shared_ptr());
    std::shared_ptr<ov::opset1::Constant> multiplyConstant;
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

    const std::shared_ptr<ov::opset1::Subtract> subtract = ov::as_type_ptr<ov::opset1::Subtract>(dataNode.get_node_shared_ptr());
    std::shared_ptr<ov::opset1::Convert> subtractConvert;
    std::shared_ptr<ov::opset1::Constant> subtractConstant;
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

    const std::shared_ptr<ov::opset1::Convert> convert = ov::as_type_ptr<ov::opset1::Convert>(dataNode.get_node_shared_ptr());
    if (convert != nullptr) {
        auto el_type = convert->input(0).get_element_type();
        auto foundIt = std::find(defaultPrecisions.begin(), defaultPrecisions.end(), el_type);
        if (foundIt == defaultPrecisions.end() &&
            el_type != element::i4  && el_type != element::u4 &&
            el_type != element::f32 && el_type != element::f16) {
            return FakeQuantizeDequantization(dataNode, nullptr, subtract, subtractConvert, subtractConstant, multiply, multiplyConstant);
        }
        dataNode = convert->get_input_source_output(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, subtractConvert, subtractConstant, multiply, multiplyConstant);
}

FakeQuantizeDequantization NetworkHelper::getDequantizationBelow(const std::shared_ptr<Node>& node, const bool convertIsMandatory) {
    const Output<Node> dataNode = node->output(0);
    const auto& targetInputs = dataNode.get_target_inputs();
    if (targetInputs.size() == 0ul) {
        return FakeQuantizeDequantization();
    }

    std::shared_ptr<Node> lastNode = targetInputs.begin()->get_node()->shared_from_this();

    const std::shared_ptr<ov::opset1::Convert> convert = ov::as_type_ptr<ov::opset1::Convert>(lastNode);
    if (convertIsMandatory && (convert == nullptr)) {
        return FakeQuantizeDequantization();
    }

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

    const std::shared_ptr<ov::opset1::Subtract> subtract = ov::as_type_ptr<ov::opset1::Subtract>(lastNode);
    std::shared_ptr<ov::opset1::Convert> subtractConvert;
    std::shared_ptr<ov::opset1::Constant> subtractConstant;
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

    const std::shared_ptr<ov::opset1::Multiply> multiply = ov::as_type_ptr<ov::opset1::Multiply>(lastNode);
    std::shared_ptr<ov::opset1::Constant> multiplyConstant;
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

    // Note: Dequantization operations might have converts between DQ constant and eltwise
    auto is_branch_const = [](const ov::Output<ov::Node>& in) {
        auto node = in.get_node_shared_ptr();
        return (ov::is_type<ov::opset1::Constant>(node)) ||
               (ov::is_type<ov::opset1::Convert>(node) && ov::is_type<ov::opset1::Constant>(node->get_input_node_shared_ptr(0)));
    };

    // Eltwise is unnormalized if it has constant 0th input and non-constant 1st input
    auto unnormalized_eltwise = [&is_branch_const](const std::shared_ptr<ov::Node>& eltwise) {
        return eltwise != nullptr && is_branch_const(eltwise->input_value(0)) && !is_branch_const(eltwise->input_value(1));
    };

    if (unnormalized_eltwise(dequantization.multiply)) {
        const auto leftParent = dequantization.multiply->input_value(0);
        const auto rightParent = dequantization.multiply->input_value(1);
        std::shared_ptr<ov::opset1::Multiply> normalized_multiply = ov::as_type_ptr<ov::opset1::Multiply>(
                dequantization.multiply->clone_with_new_inputs({rightParent, leftParent}));
        replace_node(dequantization.multiply, normalized_multiply);
        dequantization.multiply = normalized_multiply;
    }
    if (unnormalized_eltwise(dequantization.subtract)) {
        const auto leftParent = dequantization.subtract->input_value(0);
        const auto rightParent = dequantization.subtract->input_value(1);
        std::shared_ptr<ov::opset1::Subtract> normalized_subtract = ov::as_type_ptr<ov::opset1::Subtract>(
                dequantization.subtract->clone_with_new_inputs({rightParent, leftParent}));
        replace_node(dequantization.subtract, normalized_subtract);
        dequantization.subtract = normalized_subtract;
    }
    return dequantization;
}

std::shared_ptr<ov::opset1::Constant> NetworkHelper::normalizeDequantizationShape(
        const std::shared_ptr<Node>& eltwise,
        const bool convertIsExpected) {
    auto constantNode = getConstantInput(eltwise, convertIsExpected);
    if (constantNode == nullptr) {
        return nullptr;
    }

    auto constant = ov::as_type_ptr<ov::opset1::Constant>(constantNode);
    if (constant == nullptr) {
        return nullptr;
    }

    const auto getConstWithNormalizeShape = [](
        const std::shared_ptr<Node>& eltwise,
        const std::shared_ptr<ov::opset1::Constant>& constant) {
        const auto constantShape = constant->get_shape();
        if (constantShape.empty()) {
            return constant;
        }

        const size_t eltwiseRank = eltwise->get_output_partial_shape(0).rank().get_length();
        if (constantShape.size() < eltwiseRank) {
            Shape unsqueezeConstantShape(eltwiseRank - constantShape.size());
            std::iota(unsqueezeConstantShape.begin(), unsqueezeConstantShape.end(), 0ul);

            const auto newConstant = fold<ov::opset1::Unsqueeze>(
                constant->output(0),
                ov::opset1::Constant::create(element::i32, Shape{ unsqueezeConstantShape.size() }, unsqueezeConstantShape));

            return ov::as_type_ptr<ov::opset1::Constant>(newConstant);
        } else {
            return constant;
        }
    };

    const auto normalizedConstant = getConstWithNormalizeShape(eltwise, constant);
    replace_node(constant, normalizedConstant);
    copy_runtime_info(constant, normalizedConstant);

    return normalizedConstant;
}

FakeQuantizeDequantizationValues NetworkHelper::createEmptyValues(const FakeQuantizeDequantization& dequantization, const element::Type& prc) {
    const auto precision = prc.is_dynamic() ? dequantization.getPrecision() : prc;
    const std::shared_ptr<Node> multiplyConstant = dequantization.multiply ?
        dequantization.multiplyConstant->get_element_type() != precision ?
            foldConvert(dequantization.multiplyConstant->output(0), precision) :
            dequantization.multiplyConstant :
        std::make_shared<ov::opset1::Constant>(precision, Shape({}), std::vector<float>({ 1.f }));

    const std::shared_ptr<Node> subtractConstant = dequantization.subtract ?
        dequantization.subtractConstant->get_element_type() != precision ?
            foldConvert(dequantization.subtractConstant->output(0), precision) :
            dequantization.subtractConstant :
        std::make_shared<ov::opset1::Constant>(precision, Shape({}), std::vector<float>({ 0.f }));

    return FakeQuantizeDequantizationValues(subtractConstant, multiplyConstant);
}

bool NetworkHelper::isZeroConst(const std::shared_ptr<Node>& node) {
    std::shared_ptr<ov::opset1::Constant> constant = ov::as_type_ptr<ov::opset1::Constant>(node);

    if (constant == nullptr)
        return false;

    if (NetworkHelper::isScalarLike(constant)) {
        auto scalar = NetworkHelper::toScalar(constant);
        if (ov::op::util::constantIsEqualTo(scalar, 0)) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

std::shared_ptr<Node> NetworkHelper::optimizeSubtract(std::shared_ptr<ov::opset1::Subtract> subtract) {
    auto convertOnSubtract = subtract->input_value(0).get_node_shared_ptr();
    if (ov::as_type_ptr<ov::opset1::Convert>(convertOnSubtract) == nullptr) {
        return subtract;
    }

    // TODO: replace assert to condition and omit conversion part if there is no convert
    // TODO: also check convertInputType to understand if we really want to propagate type
    assert(ov::as_type_ptr<ov::opset1::Convert>(convertOnSubtract));
    const element::Type convertInputType = convertOnSubtract->get_input_element_type(0);
    const element::Type convertOutputType = convertOnSubtract->get_output_element_type(0);

    if (!convertOutputType.is_real()) {
        return subtract;
    }

    auto data = convertOnSubtract->input_value(0);
    // check if shift is a constant
    std::shared_ptr<Node> shift = subtract->get_input_node_shared_ptr(1);
    bool isShiftConstant = ov::is_type<ov::opset1::Constant>(shift);
    if (!isShiftConstant && ov::is_type<ov::opset1::Convert>(shift)) {
        // if not - we're dealing with shift->Convert (to high precision) -> Subtract
        shift = shift->get_input_node_shared_ptr(0);
        isShiftConstant = ov::is_type<ov::opset1::Constant>(shift);
    }
    if (isShiftConstant) {
        std::shared_ptr<Node> replacement;

        auto shiftConst = ov::as_type_ptr<ov::opset1::Constant>(shift);
        std::shared_ptr<ov::opset1::Constant> roundedShift;
        if (shiftConst->get_element_type() != convertInputType) {
            roundedShift = NetworkHelper::round(shiftConst, convertInputType);
        } else {
            roundedShift = shiftConst;
        }

        if (isScalarLike(roundedShift)) {
            roundedShift = toScalar(roundedShift);
            if (ov::op::util::constantIsEqualTo(roundedShift, 0)) {
                replace_node(subtract, convertOnSubtract->get_input_node_shared_ptr(0));
                roundedShift = nullptr;
            }
        }

        if (roundedShift) {
            NetworkHelper::copyInfo(shiftConst, roundedShift);

            // Propagate convertInputType down
            replacement = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(data, roundedShift->output(0));
            NetworkHelper::copyInfo(subtract, replacement);
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, convertOutputType);
            replace_node(subtract, replacement);
        }

        return replacement;
    }

    return subtract;
}

NetworkHelper::InsertDequantizationResult NetworkHelper::moveDequantizationAfter(
    const std::shared_ptr<ov::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updateOutputPrecision,
    const bool moveSubtract,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    assert(
        (NetworkHelper::getDequantization(operation, defaultPrecisions).subtractConstant == nullptr) ||
        (NetworkHelper::getDequantization(operation, defaultPrecisions).subtractConstant.get() == dequantization.subtractConstant.get()));

    assert(
        (NetworkHelper::getDequantization(operation, defaultPrecisions).multiplyConstant == nullptr) ||
        (NetworkHelper::getDequantization(operation, defaultPrecisions).multiplyConstant.get() == dequantization.multiplyConstant.get()));

    assert(operation->get_output_size() == 1);

    // we must have dequantization multiply
    assert(dequantization.multiply != nullptr);

    OPENVINO_ASSERT(operation->get_output_size() == 1,
        "moveDequantizationAfter doesn't suppport dequantization propagation for layers with several outputs. (",
        operation, " has ", operation->get_output_size(), " outputs)");

    OutputVector inputs = operation->input_values();
    const size_t dequantizationIndex = getChildInputIndex(dequantization.multiply, operation);
    inputs[dequantizationIndex] = (!moveSubtract && dequantization.subtract != nullptr) ?
        dequantization.subtract :
        dequantization.data;

    const auto newOperation = operation->clone_with_new_inputs(inputs);
    newOperation->set_friendly_name(operation->get_friendly_name());
    ov::copy_runtime_info(operation, newOperation);

    if (const auto op = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(newOperation)) {
        op->set_overridden_output_type(updateOutputPrecision ?
            newOperation->get_input_element_type(0) :
            dequantization.multiplyConstant->get_element_type());
        newOperation->validate_and_infer_types();
    } else {
        OPENVINO_ASSERT(updateOutputPrecision, "moveDequantizationAfter can't save old output precision since layer is not TypeRelaxed: ", newOperation);
    }

    std::shared_ptr<Node> parent = newOperation;

    const element::Type deqPrecision = dequantization.multiplyConstant->get_element_type();
    const bool shouldConvert = (newOperation->get_output_element_type(0) != deqPrecision);
    if (shouldConvert) {
        const auto convertOutputPrecision = dequantization.convert ? dequantization.convert->get_element_type() : deqPrecision;
        parent = std::make_shared<ov::opset1::Convert>(parent, convertOutputPrecision);
        ov::copy_runtime_info({ newOperation, parent }, parent);
    }

    if (moveSubtract && (dequantization.subtract != nullptr)) {
        if (dequantization.subtractConvert == nullptr) {
            const element::Type parentPrecision = parent->get_output_element_type(0);
            if (parentPrecision.bitwidth() < dequantization.subtractConstant->get_element_type().bitwidth()) {
                THROW_IE_LPT_EXCEPTION(*parent) <<
                    "unexpected precisions: on data " << parent->get_friendly_name() << ":" << parentPrecision <<
                    ", subtract dequantization constant " << dequantization.subtractConstant->get_friendly_name() << ":" <<
                    dequantization.subtractConstant->get_element_type();
            }

            parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
                element::TypeVector{ element::f32, element::f32 }, element::TypeVector{ element::f32 },
                ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(foldConvert(dequantization.subtractConstant, parentPrecision), element::f32).get());
            ov::copy_runtime_info({ newOperation, parent }, parent);
        } else {
            // Subtract constant could be changed (including a shape) before propagation in some cases
            // so it's necessary to compute the shape for a subtractConvert before creating a new subtract
            dequantization.subtractConvert->validate_and_infer_types();
            parent = std::make_shared<ov::opset1::Subtract>(parent, dequantization.subtractConvert);
            ov::copy_runtime_info({ newOperation, parent }, parent);
        }
    }

    if (dequantization.multiply != nullptr) {
        const element::Type parentPrecision = parent->get_output_element_type(0);
        if (parentPrecision.bitwidth() < dequantization.multiplyConstant->get_element_type().bitwidth()) {
            THROW_IE_LPT_EXCEPTION(*parent) <<
                "unexpected precisions: on data " << parent->get_friendly_name() << ":" << parentPrecision <<
                ", multiply dequantization constant " << dequantization.multiplyConstant->get_friendly_name() << ":" <<
                dequantization.multiplyConstant->get_element_type();
        }

        parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            ov::opset1::Multiply(parent, foldConvert(dequantization.multiplyConstant, parentPrecision)),
            dequantization.multiply->get_output_element_type(0));
        ov::copy_runtime_info({ newOperation, parent }, parent);
    }

    insertDequantizationAfter(operation, parent, newOperation);

    if ((!moveSubtract) && (dequantization.convert != nullptr) && (dequantization.subtract != nullptr)) {
        // issue #43088
        // NetworkHelper::optimizeElementwise(dequantization.subtract);
    }

    return InsertDequantizationResult(newOperation, parent);
}

NetworkHelper::InsertDequantizationResult NetworkHelper::moveDequantizationBefore(
    const std::shared_ptr<ov::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool moveSubtract) {
    assert(
        (NetworkHelper::getDequantizationBelow(operation).subtractConstant == nullptr) ||
        (NetworkHelper::getDequantizationBelow(operation).subtractConstant.get() == dequantization.subtractConstant.get()));

    assert(
        (NetworkHelper::getDequantizationBelow(operation).multiplyConstant == nullptr) ||
        (NetworkHelper::getDequantizationBelow(operation).multiplyConstant.get() == dequantization.multiplyConstant.get()));
    std::vector<std::vector<std::shared_ptr<ov::opset1::Constant>>> multiplyConstants, subtractConstants;
    if (is_type<ov::opset1::Concat>(operation)) {
        const auto concatNode = as_type_ptr<ov::opset1::Concat>(operation);
        int64_t axis = -1;
        if (concatNode->get_output_partial_shape(0).rank().is_static()) {
            const auto rank = concatNode->get_output_partial_shape(0).rank().get_length();
            axis = ov::util::normalize(concatNode->get_axis(), rank);
        }
        if (dequantization.multiply && dequantization.multiplyConstant->get_shape().size() > 1 && dequantization.multiplyConstant->get_shape()[axis] != 1) {
            multiplyConstants = NetworkHelper::splitConstantsBeforeConcat(operation, { dequantization.multiplyConstant });
        }
        if (dequantization.subtract && dequantization.subtractConstant->get_shape().size() > 1 && dequantization.subtractConstant->get_shape()[axis] != 1) {
            subtractConstants = NetworkHelper::splitConstantsBeforeConcat(operation, { dequantization.subtractConstant });
        }
    } else {
        multiplyConstants = {{ dequantization.multiplyConstant }};
        subtractConstants = {{ dequantization.subtractConstant }};
    }
    std::vector<std::shared_ptr<ov::Node>> newNodes;
    for (size_t i = 0; i < operation->get_input_size(); ++i) {
        auto parent = operation->get_input_node_shared_ptr(i);
        const element::Type deqPrecision = dequantization.multiplyConstant->get_element_type();
        const bool shouldConvert = (operation->get_output_element_type(0) != deqPrecision);
        if (shouldConvert) {
            const auto convertOutputPrecision = dequantization.convert != nullptr ?
                dequantization.convert->get_output_element_type(0) :
                deqPrecision;
            parent = std::make_shared<ov::opset1::Convert>(parent, convertOutputPrecision);
            if (dequantization.convert == nullptr) {
                THROW_TRANSFORMATION_EXCEPTION << "dequantization convert is absent";
            }

            parent->set_friendly_name(dequantization.convert->get_friendly_name() + "_" + std::to_string(i + 1));
            ov::copy_runtime_info(dequantization.convert, parent);
        }
        if (moveSubtract && (dequantization.subtract != nullptr)) {
            if (dequantization.subtractConvert == nullptr) {
                const element::Type parentPrecision = parent->get_output_element_type(0);
                if (parentPrecision.bitwidth() < dequantization.subtractConstant->get_element_type().bitwidth()) {
                    THROW_IE_LPT_EXCEPTION(*parent) <<
                        "unexpected precisions: on data " << parent->get_friendly_name() << ":" << parentPrecision <<
                        ", subtract dequantization constant " << dequantization.subtractConstant->get_friendly_name() << ":" <<
                        dequantization.subtractConstant->get_element_type();
                }
                auto subtractConstant = subtractConstants.size() ? subtractConstants[0][i] : dequantization.subtractConstant;
                parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
                    std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{ element::f32 },
                    ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(
                        subtractConstant->output(0).get_element_type() == parentPrecision ?
                        subtractConstant :
                        foldConvert(subtractConstant, parentPrecision), element::f32).get());
                parent->set_friendly_name(dequantization.subtract->get_friendly_name() + "_" + std::to_string(i + 1));
            } else {
                // Subtract constant could be changed (including a shape) before propagation in some cases
                // so it's necessary to compute the shape for a subtractConvert before creating a new subtract
                dequantization.subtractConvert->validate_and_infer_types();
                parent = std::make_shared<ov::opset1::Subtract>(parent, dequantization.subtractConvert);
            }
            ov::copy_runtime_info(dequantization.subtract, parent);
        }

        if (dequantization.multiply != nullptr) {
            auto multiplyConstant = multiplyConstants.size() ? multiplyConstants[0][i] : dequantization.multiplyConstant;
            const element::Type parentPrecision = parent->get_output_element_type(0);
            if (parentPrecision.bitwidth() < multiplyConstant->get_element_type().bitwidth()) {
                THROW_IE_LPT_EXCEPTION(*parent) <<
                    "unexpected precisions: on data " << parent->get_friendly_name() << ":" << parentPrecision <<
                    ", multiply dequantization constant " << multiplyConstant->get_friendly_name() << ":" << multiplyConstant->get_element_type();
            }

            parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                ov::opset1::Multiply(parent,
                    multiplyConstant->output(0).get_element_type() == parentPrecision ?
                    multiplyConstant :
                    foldConvert(multiplyConstant->output(0), parentPrecision)),
                dequantization.multiply->get_output_element_type(0));
            ov::copy_runtime_info(dequantization.multiply, parent);
            parent->set_friendly_name(dequantization.multiply->get_friendly_name() + "_" + std::to_string(i + 1));
        }
        if ((!moveSubtract) && (dequantization.convert != nullptr) && (dequantization.subtract != nullptr)) {
            // issue #43088
            // NetworkHelper::optimizeElementwise(dequantization.subtract);
        }
        newNodes.push_back(parent);
    }
    auto newOperation = operation->clone_with_new_inputs(ov::OutputVector(newNodes.begin(), newNodes.end()));
    NetworkHelper::copyInfo(operation, newOperation);
    if (dequantization.multiply == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "dequantization operations must end with multiply";
    }
    replace_node(dequantization.multiply, newOperation);
    if (const auto op = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(newOperation)) {
        op->set_overridden_output_type(dequantization.multiplyConstant->get_element_type());
        newOperation->validate_and_infer_types();
    }

    return InsertDequantizationResult(newOperation, dequantization.multiply);
}

std::vector<std::vector<std::shared_ptr<ov::opset1::Constant>>> NetworkHelper::splitConstantsBeforeConcat(const std::shared_ptr<ov::Node> concat,
    const std::vector<std::shared_ptr<ov::opset1::Constant>> currConstants) {
    std::vector<std::vector<std::shared_ptr<ov::opset1::Constant>>> newConstants(currConstants.size());
    auto number_of_concat_inputs = concat->get_input_size();
    const auto concatNode = as_type_ptr<ov::opset1::Concat>(concat);
        int64_t concat_axis = -1;
        if (concatNode->get_output_partial_shape(0).rank().is_static()) {
            const auto rank = concatNode->get_output_partial_shape(0).rank().get_length();
            concat_axis = ov::util::normalize(concatNode->get_axis(), rank);
        }
    std::vector<int64_t> shape_axis(number_of_concat_inputs);
    for (size_t i{ 0 }; i < number_of_concat_inputs; ++i) {
        auto shape = concat->get_input_partial_shape(i);
        shape_axis[i] = shape[concat_axis].get_length();
    }
    for (size_t i = 0; i < currConstants.size(); ++i) {
        std::vector<std::shared_ptr<ov::opset1::Constant>> newConstant;
        const auto const_shape = currConstants[i]->get_shape();
        if (ov::shape_size(const_shape) == 1 || const_shape[concat_axis] == 1) {
            newConstant.push_back(currConstants[i]);
            newConstants[i] = newConstant;
            continue;
        }
        auto split = std::make_shared<ov::opset1::VariadicSplit>(currConstants[i],
            ov::opset1::Constant::create(element::i64, Shape{}, { concat_axis }),
            ov::opset1::Constant::create(element::i64, Shape{ number_of_concat_inputs }, shape_axis));
        OutputVector outputResults(split->get_output_size());
        auto foldResult = split->constant_fold(outputResults, split->input_values());
        if (!foldResult) {
            THROW_IE_LPT_EXCEPTION(*concat) << "error when splitting constants before concat " <<
                concat->get_friendly_name();
        }
        for (auto outputResult : outputResults) {
            auto constant = as_type_ptr<ov::opset1::Constant>(outputResult.get_node_shared_ptr());
            newConstant.push_back(constant);
        }

        newConstants[i] = newConstant;
    }
    return newConstants;
}

bool NetworkHelper::checkConstantValuePrecision(const element::Type expectedPrecision, const std::shared_ptr<Node>& constant) {
    if (expectedPrecision.is_signed()) {
        return true;
    }

    std::shared_ptr<ov::opset1::Constant> constantOp = ov::as_type_ptr<ov::opset1::Constant>(constant);
    if (constantOp == nullptr) {
        return false;
    }

    const auto values = constantOp->cast_vector<float>();
    const bool convertCanBeRemoved =
        (expectedPrecision.is_signed() || (std::all_of(values.begin(), values.end(), [](const float value) { return value >= 0.f; })));
    return convertCanBeRemoved;
}

size_t NetworkHelper::getChildInputIndex(const std::shared_ptr<ov::Node>& parent, const std::shared_ptr<ov::Node>& child) {
    for (size_t i = 0; i < child->get_input_size(); ++i) {
        if (parent.get() == child->get_input_node_ptr(i)) {
            return i;
        }
    }
    THROW_IE_LPT_EXCEPTION(*child) << "child input index between " <<
        parent->get_friendly_name() << " and " << child->get_friendly_name() << " was not found";
}

size_t NetworkHelper::getParentOutputIndex(const std::shared_ptr<ov::Node>& parent, const std::shared_ptr<ov::Node>& child) {
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
    std::shared_ptr<ov::opset1::Constant> constant = ov::as_type_ptr<ov::opset1::Constant>(node);
    if (constant == nullptr) {
        return node;
    }

    if (!NetworkHelper::isScalarLike(constant)) {
        return node;
    }

    return NetworkHelper::toScalar(constant);
}

std::shared_ptr<Node> foldConvert(const Output<Node>& node, const element::Type targetPrecision) {
    if (ov::is_type<ov::opset1::Constant>(node.get_node_shared_ptr()) && (node.get_element_type() == targetPrecision)) {
        return node.get_node_shared_ptr();
    }

    return fold<ov::opset1::Convert>(node, targetPrecision);
}

bool NetworkHelper::checkZeroPoint(const std::shared_ptr<Node>& node, const DataPrecision& dataPrecision) {
    if (!node) {
        return true;
    }

    float min, max;
    if (ov::is_type<ov::opset1::Subtract>(node)) {
        const auto parent = node->get_input_node_shared_ptr(0);
        const auto intNode = ov::is_type<ov::opset1::Convert>(parent) ? parent : node;
        const auto type = intNode->get_input_element_type(0);
        if (type == element::u8 || type == element::i8) {
            min = DataPrecision::getMinValue(type, levels::int8) - 0.5f;
            max = DataPrecision::getMaxValue(type, levels::int8) + 0.5f;
        } else {
            return type == element::f32 || type == element::f16;
        }
        auto subtract1input = node->get_input_node_shared_ptr(1);
        if (ov::is_type<ov::opset1::Convert>(subtract1input)) {
            return true;
        }
        auto subtractConst = ov::as_type_ptr<ov::opset1::Constant>(subtract1input);
        if (!subtractConst) {
            subtractConst = ov::as_type_ptr<ov::opset1::Constant>(node->get_input_node_shared_ptr(1)->get_input_node_shared_ptr(0));
            if (subtractConst == nullptr) {
                return false;
            }
        }
        const auto subtractValues = subtractConst->cast_vector<float>();
        if (std::any_of(subtractValues.begin(), subtractValues.end(), [min, max](const float& val) {
            return (val < min) || (val > max); })) {
            return false;
        }
    } else if (ov::is_type<ov::opset1::FakeQuantize>(node)) {
        if (!dataPrecision.hasZeroPoint) {
            return true;
        }
        min = dataPrecision.min - 0.5f;
        max = dataPrecision.max + 0.5f;
        const auto quantizationDetails = QuantizationDetails::getDetails(ov::as_type_ptr<ov::opset1::FakeQuantize>(node));
        for (size_t i = 0; i < quantizationDetails.outputLowValues.size(); ++i) {
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

    for (auto i : v1) {
        for (auto j : v2) {
            if (i == j) {
                v3.push_back(i);
                break;
            }
        }
    }

    return v3;
}

bool isDisabled(const std::shared_ptr<Node>& node) {
    for (const auto& input : node->inputs()) {
        auto precisionAttribute = getAttribute<PrecisionsAttribute>(input);
        if (precisionAttribute.empty()) {
            continue;
        }
        const auto& precisionRestrictions = precisionAttribute.as<PrecisionsAttribute>().value();
        if (precisionRestrictions.empty()) {
            return true;
        }
    }
    return false;
}

void NetworkHelper::insertDequantizationAfter(
    const std::shared_ptr<Node>& originalNode,
    const std::shared_ptr<Node>& dequantization,
    const std::shared_ptr<Node>& newNode) {
    replace_node(originalNode, dequantization);

    // We do it to avoid dequantization propagation to the shapeOf subgraphs
    for (const auto& input : dequantization->get_output_target_inputs(0)) {
        if (const auto shapeOf = as_type_ptr<ov::opset1::ShapeOf>(input.get_node()->shared_from_this())) {
            const auto newShapeOf = shapeOf->clone_with_new_inputs({ newNode });
            replace_node_update_name(shapeOf, newShapeOf);
        }
    }
}

ov::Output<ov::Node> NetworkHelper::getSingleConsumerConstant(const ov::Output<ov::Node>& output) {
    const auto node = output.get_node();
    if (!ov::is_type<ov::opset1::Constant>(node))
        THROW_IE_LPT_EXCEPTION(*node) << "getSingleConsumerConstant Expected Constant node type";
    return output.get_target_inputs().size() == 1
        ? output
        : node->clone_with_new_inputs(node->input_values())->output(0);
}

bool NetworkHelper::checkConstantNotInf(const std::shared_ptr<Node> constant_node) {
    const auto constant = ov::as_type_ptr<ov::opset1::Constant>(constant_node);
    if (constant == nullptr)
        return false;
    const auto values = constant->cast_vector<float>();
    return std::all_of(values.begin(), values.end(), [](const float x) { return !std::isinf(x); });
}
} // namespace low_precision
} // namespace pass
} // namespace ov
