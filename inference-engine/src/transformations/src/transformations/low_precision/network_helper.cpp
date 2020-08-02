// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/network_helper.hpp>

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

#include <ngraph/rt_info.hpp>
#include <transformations/low_precision/common/ie_lpt_exception.hpp>

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

int NetworkHelper::onWeightsInDepth(std::shared_ptr<Node> layer) {
    const std::vector<std::shared_ptr<Node>> children = consumers(layer);
    for (std::shared_ptr<Node> child : children) {
        if ((is_type<opset1::Convolution>(child) || is_type<opset1::GroupConvolution>(child) || is_type<opset1::MatMul>(child)) &&
            (child->inputs().size() >= 2lu)) {
            const std::vector<std::shared_ptr<Node>> parents = getParentsRecursivelyExceptTypes(child, {}, 1);
            for (const std::shared_ptr<Node>& parent : parents) {
                if (parent.get() == layer.get()) {
                    return 1;
                }
            }
            return -1;
        }

        const int result = onWeightsInDepth(child);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

bool NetworkHelper::onWeights(std::shared_ptr<Node> layer) {
    const int result = onWeightsInDepth(layer);
    return result == 1;
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
    size_t i = 0ul;
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
    auto multiply = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch);

    auto multiplyParent1 = multiply->get_input_node_shared_ptr(0);
    auto multiplyParent2 = multiply->get_input_node_shared_ptr(1);

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

    auto x = multiply->get_input_node_shared_ptr(multiplyInputBranch);
    auto a = multiply->get_input_node_shared_ptr(multiplyInputBranch == 0 ? 1 : 0);
    auto b = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch == 0 ? 1 : 0);
    auto bDivA = fold<opset1::Divide>(b, a);

    auto newAdd = std::make_shared<op::TypeRelaxed<opset1::Add>>(
        opset1::Add(x, bDivA),
        addAfterMultiply->get_output_element_type(multiplyBranch));

    auto newMultiply = std::make_shared<opset1::Multiply>(newAdd, a);
    replace_node(addAfterMultiply, newMultiply);

    copyInfo(addAfterMultiply, newAdd);
    return newMultiply;
}

bool NetworkHelper::isScalarLike(std::shared_ptr<opset1::Constant> constant) {
    return constant->get_all_data_elements_bitwise_identical();
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

std::shared_ptr<ngraph::opset1::Multiply> NetworkHelper::optimizeMultipliesAfter(std::shared_ptr<Node> node) {
    std::shared_ptr<ngraph::opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(node);

    if (multiply && multiply->output(0).get_target_inputs().size() == 1) {
        auto constant1 = getConstantInput(multiply);
        if (!constant1 || constant1->output(0).get_target_inputs().size() != 1) {
            return multiply;
        }
        auto nextMultiplyInput = *multiply->output(0).get_target_inputs().begin();
        auto nextMultiply = as_type_ptr<opset1::Multiply>(nextMultiplyInput.get_node()->shared_from_this());
        if (nextMultiply) {
            auto constant2 = getConstantInput(nextMultiply);
            auto constant2Inputs = constant2->output(0).get_target_inputs().size();
            if (!constant2 || constant2->output(0).get_target_inputs().size() != 1) {
                return multiply;
            }

            auto newConst = fold<opset1::Multiply>(constant1, constant2);
            auto newMultiply =
                    std::make_shared<opset1::Multiply>(
                            multiply->input_value(1 - constant1->output(0).get_target_inputs().begin()->get_index()),
                            newConst->output(0));
            replace_node(nextMultiply, newMultiply);
            return newMultiply;
        }
    }

    return nullptr;
}

std::shared_ptr<opset1::Constant> NetworkHelper::roundWithTolerance(std::shared_ptr<Node> node, element::Type target_type, float tolerance) {
    auto constant = as_type_ptr<opset1::Constant>(node);
    assert(constant);
    auto values = constant->cast_vector<float>();

    auto castedConstant = as_type_ptr<opset1::Constant>(fold<opset1::Convert>(constant, target_type));
    auto castedValues = castedConstant->cast_vector<float>();

    // TODO: implement with constant folding when ReduceAnd constant folding is ready
    if (std::equal(values.begin(), values.end(), castedValues.begin(), [tolerance](float a, float b) { return fabs(a - b) < tolerance; })) {
        return castedConstant;
    } else {
        return constant;
    }
}

// Decompose FakeQuantize to FakeQuantize with output integer limits (quantize), dequatized MultiplyAdd
// To align types the resulting sequence is FakeQuantize -> Convert -> Convert -> MultiplyAdd
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> NetworkHelper::decomposeFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    const element::Type precision,
    const float min,
    const float max,
    const bool hasZeroPoint,
    const bool updatePrecision) {
    using std::make_shared;

    const auto newMin = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    const auto newMax = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    const auto outputLow = fq->input_value(3);
    const auto outputHigh = fq->input_value(4);

    // TODO: threshold values have to used here to avoid shifts
    const std::shared_ptr<Node> scale = fold<opset1::Divide>(
        fold<opset1::Subtract>(outputHigh, outputLow),
        fold<opset1::Subtract>(newMax, newMin));

    std::shared_ptr<Node> shift = hasZeroPoint ?
        fold<opset1::Divide>(
            fold<opset1::Subtract>(fold<opset1::Multiply>(newMin, outputHigh), fold<opset1::Multiply>(newMax, outputLow)),
            fold<opset1::Subtract>(outputHigh, outputLow)) :
        nullptr;

    // Build a substitution sub-graph:

    std::shared_ptr<ngraph::Node> newFQ = fold_fake_quantize<opset1::FakeQuantize>(
            fq->input_value(0),
            fq->input_value(1),
            fq->input_value(2),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast());
    // TODO: for debuging only - remove later
    newFQ->set_friendly_name(fq->get_friendly_name() + "_original");

    if (shift != nullptr) {
        std::shared_ptr<opset1::Constant> shiftConst = as_type_ptr<opset1::Constant>(shift);
        if (isScalarLike(shiftConst)) {
            auto scalar = toScalar(shiftConst);
            if (op::util::constantIsEqualTo(scalar, 0)) {
                shift = nullptr;
            }
        }
    }

    std::shared_ptr<ngraph::Node> convert2;
    if (updatePrecision) {
        std::shared_ptr<Node> convert;
        std::shared_ptr<opset1::Constant> newFqConstant = as_type_ptr<opset1::Constant>(newFQ);

        if (is_type<opset1::Constant>(newFQ)) {
            convert = fold<opset1::Convert>(newFQ, precision);
        } else if (is_type<opset1::FakeQuantize>(newFQ)) {
            newFQ = setOutDataPrecision(as_type_ptr<opset1::FakeQuantize>(newFQ), precision);
            auto type = newFQ->get_output_element_type(0);
            convert = newFQ;
        } else {
            THROW_IE_LPT_EXCEPTION(*newFQ) << "unexpected operation type";
        }

        auto pre = fq->get_output_element_type(0);
        convert2 = make_shared<opset1::Convert>(convert, fq->get_output_element_type(0));
    }

    std::shared_ptr<ngraph::Node> sub = shift == nullptr ? nullptr :
        make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
            convert2 == nullptr ? newFQ : convert2,
            shift);

    // const auto subShape = sub->get_shape();
    // const auto scaleShape = scale->get_shape();
    // auto dequantize = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Multiply>>(
    auto dequantize = make_shared<ngraph::opset1::Multiply>(
        sub == nullptr ?
            convert2 == nullptr ? newFQ : convert2:
            sub,
        scale);

    auto outputs = newFQ->get_outputs();
    //NetworkHelper::setOutDataPrecision(newFQ, precision);

    // sub->set_overriden_output_type(element::f32);
    // std::dynamic_pointer_cast<ngraph::Node>(sub)->validate_and_infer_types();

    replace_node(fq, dequantize);
    // Make type-relaxed node

    return std::make_tuple(newFQ, dequantize);
}

std::shared_ptr<opset1::FakeQuantize> NetworkHelper::updateFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max) {
    auto newMin = std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newMax = std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    // TODO: question: why we need fold?
    // auto newFQ = fold_fake_quantize<opset1::FakeQuantize>(
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
    const float dequantizationScale,
    const float dequantizationShift,
    const ngraph::element::Type originalPrecision,
    const ngraph::Shape dataNodeOutputShape,
    element::Type precision,
    float min,
    float max) {
    // TODO: we create input here! we really need it here?
    const std::shared_ptr<opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(precision, dataNodeOutputShape);
    std::shared_ptr<ngraph::Node> parent = input;

    // TODO: convert should be optional: where is updatePrecision?
    std::shared_ptr<ngraph::opset1::Convert> convert;
    {
        convert = as_type_ptr<ngraph::opset1::Convert>(fold<opset1::Convert>(
            input,
            originalPrecision));
        parent = convert;
    }

    std::shared_ptr<ngraph::opset1::Subtract> subtract;
    if (dequantizationShift != 0.f) {
        subtract = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalPrecision, ngraph::Shape({}), std::vector<float>({ dequantizationShift })));
        subtract->set_output_type(0, originalPrecision, subtract->get_output_partial_shape(0));
        parent = subtract;
    }

    // mandatory
    std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(originalPrecision, ngraph::Shape({}), std::vector<float>({ dequantizationScale })));

    return FakeQuantizeDequantization(input, convert, subtract, multiply);
}

FakeQuantizeDequantization NetworkHelper::createDequantizationFromFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max) {
    using std::make_shared;

    const ngraph::element::Type_t fqPrecision = fq->get_output_element_type(0);
    auto newMin = make_shared<opset1::Constant>(fqPrecision, Shape{}, min);
    auto newMax = make_shared<opset1::Constant>(fqPrecision, Shape{}, max);

    auto outputLow = fq->input_value(3);
    auto outputHigh = fq->input_value(4);

    // TODO: threshold values have to used here to avoid shifts

    const std::shared_ptr<Node> scale = fold<opset1::Divide>(
        fold<opset1::Subtract>(outputHigh, outputLow),
        fold<opset1::Subtract>(newMax, newMin));

    std::shared_ptr<Node> shift = fold<opset1::Divide>(
        fold<opset1::Subtract>(fold<opset1::Multiply>(newMin, outputHigh), fold<opset1::Multiply>(newMax, outputLow)),
        fold<opset1::Subtract>(outputHigh, outputLow));
    Shape shiftShape = shift->get_output_shape(0);

    const opset1::Constant* shiftConstant = as_type<opset1::Constant>(shift.get());
    if (shiftConstant->get_all_data_elements_bitwise_identical()) {
        const std::vector<float> values = shiftConstant->cast_vector<float>();
        if (values[0] == 0.0f) {
            shift = nullptr;
        }
    }

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, fq->get_output_shape(0));
    const std::shared_ptr<ngraph::opset1::Convert> convert =  as_type_ptr<ngraph::opset1::Convert>(fold<opset1::Convert>(
        input,
        fq->get_output_element_type(0)));

    const std::shared_ptr<ngraph::opset1::Subtract> subtract = shift == nullptr ?
        nullptr :
        make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(convert, shift);
    if (subtract != nullptr) {
        subtract->set_output_type(0, fq->get_output_element_type(0), subtract->get_output_partial_shape(0));
    }

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = make_shared<ngraph::opset1::Multiply>(
        subtract == nullptr ? static_cast<std::shared_ptr<Node>>(convert) : subtract,
        scale);

    return FakeQuantizeDequantization(fq, convert, subtract, multiply);
}

FakeQuantizeDequantization NetworkHelper::getDequantization(const std::shared_ptr<Node> node, const size_t parentIndex) {
    std::shared_ptr<Node> dataNode = node->input_value(parentIndex).get_node_shared_ptr();

    const std::shared_ptr<ngraph::opset1::Multiply> multiply =
        (dataNode->get_input_size() > 1ul) && is_type<opset1::Constant>(dataNode->get_input_node_ptr(1)) ?
        as_type_ptr<ngraph::opset1::Multiply>(dataNode) :
        nullptr;
    if (multiply != nullptr) {
        dataNode = multiply->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Subtract> subtract = (dataNode->get_input_size() > 1ul) && is_type<opset1::Constant>(dataNode->get_input_node_ptr(1)) ?
        as_type_ptr<opset1::Subtract>(dataNode) :
        nullptr;
    if (subtract != nullptr) {
        dataNode = subtract->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode);
    if (convert != nullptr) {
        dataNode = convert->get_input_node_shared_ptr(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, multiply);
}

FakeQuantizeDequantizationValues NetworkHelper::createEmptyValues(const FakeQuantizeDequantization& dequantization) {
    std::shared_ptr<Node> parent = dequantization.convert ? dequantization.convert : dequantization.data;

    std::shared_ptr<Node> multiply1Const = dequantization.multiply ?
        dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 1.f }));

    std::shared_ptr<Node> subtract1Const = dequantization.subtract ?
        dequantization.subtract->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
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

std::vector<std::shared_ptr<Node>> NetworkHelper::getChildrenRecursivelyExceptTypes(
    const std::shared_ptr<Node>& layer,
    const std::unordered_set<std::string>& exceptionLayerTypes) {
    std::queue<std::shared_ptr<Node>> notHandledChildren;

    for (const ngraph::descriptor::Output& output : layer->get_outputs()) {
        for (const ngraph::descriptor::Input* input : output.get_inputs()) {
            std::shared_ptr<Node> child = input->get_node();
            notHandledChildren.emplace(child);
        }
    }

    std::vector<std::shared_ptr<Node>> resultChildren;

    while (!notHandledChildren.empty()) {
        const std::shared_ptr<ngraph::Node> operation = notHandledChildren.front();
        notHandledChildren.pop();

        const std::string typeName = operation->get_type_name();
        if (exceptionLayerTypes.find(typeName) == exceptionLayerTypes.end()) {
            resultChildren.push_back(operation);
            continue;
        }

        for (auto output : operation->get_outputs()) {
            for (auto input : output.get_inputs()) {
                std::shared_ptr<Node> child = input->get_node();
                notHandledChildren.emplace(child);
            }
        }
    }

    return resultChildren;
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

    auto data = convertOnSubtract->input_value(0);
    auto shift = subtract->input_value(1).get_node_shared_ptr();
    auto roundedShift = NetworkHelper::roundWithTolerance(shift, convertInputType);

    std::shared_ptr<Node> replacement;
    if (roundedShift->get_element_type() == convertInputType) {
        // Propagate convertInputType down
        // replacement = std::make_shared<opset1::Subtract>(data, roundedShift);
        replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(data, roundedShift);
        replacement->set_output_type(0, convertOutputType, replacement->get_output_partial_shape(0));
        replace_node(subtract, replacement);
    } else {
        // Try to represent it as data - (-b)
        roundedShift = NetworkHelper::roundWithTolerance(fold<opset1::Negative>(shift), convertInputType);
        if (roundedShift->get_element_type() == convertInputType) {
            // Assuming Subtract will go out of representable set of values for target type
            // So keep the original data type (likely not integer)
            replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
                    opset1::Subtract(data, roundedShift),
                    convertOnSubtract->get_output_element_type(0));
            replacement->set_output_type(0, convertOutputType, replacement->get_output_partial_shape(0));
            replace_node(subtract, replacement);
        }
    }

    // We lose the tail conversion here; not needed if the next node is a TypeRelaxed
    // TODO: check cases when Convert should be preserved

    // Try to optimize Add out if constant is zero
    // TODO: don't remove operation here: don't create this Subtraction operation in FQ decomposition
    // if (isScalarLike(roundedShift)) {
    //    auto scalar = distillToScalar(roundedShift);
    //    if (op::util::constantIsEqualTo(scalar, 0)) {
    //        replace_node(replacement, replacement->input_value(0).get_node_shared_ptr());
    //        replacement = nullptr;
    //    }
    // }

    return replacement;
}

// TODO: don't use it
void NetworkHelper::moveDequantization(
    const std::shared_ptr<ngraph::Node> operation,
    const std::shared_ptr<ngraph::Node> dequantization,
    const std::shared_ptr<ngraph::Node> scalesConst,
    const std::shared_ptr<ngraph::Node> shiftsConst) {
    const std::shared_ptr<Node> dequantizationParent = dequantization->input_value(0).get_node_shared_ptr();

    // TODO: refactor: the second operation input is passed obviously
    auto newOperation =
        operation->get_input_size() == 1ul ?
        operation->copy_with_new_inputs({ dequantizationParent }) :
        operation->copy_with_new_inputs({ dequantizationParent, operation->input_value(1).get_node_shared_ptr() });

    const auto newDequantization = dequantization->copy_with_new_inputs({
        newOperation,
        scalesConst == nullptr ? dequantization->input_value(1) : scalesConst,
        shiftsConst == nullptr ? dequantization->input_value(2) : shiftsConst });

    const std::string friendlyName = operation->get_friendly_name();
    // TODO: new operation name has to be unique
    newOperation->set_friendly_name(friendlyName + "_original");
    newDequantization->set_friendly_name(friendlyName);

    replace_node(operation, newDequantization);
}

NetworkHelper::InsertDequantizationResult NetworkHelper::moveDequantizationAfter(
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updatePrecision) {
    std::vector<Output<Node>> inputs(operation->get_input_size());
    for (size_t i = 0; i < operation->get_input_size(); ++i) {
        inputs[i] = operation->get_input_node_shared_ptr(i);
    }

    const size_t dequantizationIndex = getInputIndex(dequantization.multiply, operation);
    inputs[dequantizationIndex] = dequantization.data;

    std::shared_ptr<ngraph::Node> newOperation = operation->clone_with_new_inputs(inputs);
    newOperation->set_friendly_name(operation->get_friendly_name());
    // copyInfo(operation, newOperation);

    const bool shouldConvert = (newOperation->get_output_element_type(0) != dequantization.multiply->get_output_element_type(0));
    const std::shared_ptr<ngraph::opset1::Convert> convert = (updatePrecision || shouldConvert) ? dequantization.convert : nullptr;

    auto parent = newOperation;
    if (convert != nullptr) {
        parent = std::make_shared<opset1::Convert>(parent, convert->get_output_element_type(0));
    }
    if (dequantization.subtract != nullptr) {
        auto subtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);
        parent = std::make_shared<opset1::Subtract>(parent, subtractConstant);
    }
    if (dequantization.multiply != nullptr) {
        auto multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);
        parent = std::make_shared<opset1::Multiply>(parent, multiplyConstant);
    }
    replace_node(operation, parent);

    if (updatePrecision) {
        NetworkHelper::setOutDataPrecision(newOperation, newOperation->get_input_element_type(0));
    }

    return InsertDequantizationResult(newOperation, parent);
}

NetworkHelper::InsertDequantizationResult NetworkHelper::moveMultiplyAfter(
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool removeConvert) {
    std::vector<Output<Node>> inputs(operation->get_input_size());
    for (size_t i = 0; i < operation->get_input_size(); ++i) {
        inputs[i] = operation->get_input_node_shared_ptr(i);
    }

    const size_t dequantizationIndex = getInputIndex(dequantization.multiply, operation);
    inputs[dequantizationIndex] = dequantization.multiply->get_input_node_shared_ptr(0);

    std::shared_ptr<ngraph::Node> newOperation = operation->clone_with_new_inputs(inputs);
    newOperation->set_friendly_name(operation->get_friendly_name());

    auto multiplyReplacement = as_type_ptr<opset1::Multiply>(dequantization.multiply->clone_with_new_inputs({
        newOperation,
        dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) }));
    replace_node(operation, multiplyReplacement);

    if (removeConvert && (dequantization.convert != nullptr)) {
        if (dequantization.subtract == nullptr) {
            inputs[dequantizationIndex] = dequantization.convert->get_input_node_shared_ptr(0);
            auto newOperation2 = newOperation->clone_with_new_inputs(inputs);
            replace_node(newOperation, newOperation2);
            newOperation = newOperation2;
        } else {
            removeConvertIfPossible(newOperation, dequantization);
        }
    }

    return InsertDequantizationResult(newOperation, multiplyReplacement);
}

void NetworkHelper::removeConvertIfPossible(
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization) {
    const auto convertParent = dequantization.convert->get_input_node_shared_ptr(0);
    const size_t convertIndex = getInputIndex(convertParent, dequantization.convert);
    const element::Type precisionBeforeConvert = convertParent->get_output_element_type(convertIndex);

    if (checkConstantValuePrecision(precisionBeforeConvert, dequantization.subtract->get_input_node_shared_ptr(1))) {
        auto newSubtract = dequantization.subtract->clone_with_new_inputs({
            dequantization.convert->get_input_node_shared_ptr(0),
            fold<opset1::Convert>(dequantization.subtract->get_input_node_shared_ptr(1), precisionBeforeConvert) });
        replace_node(dequantization.subtract, newSubtract);
    }
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

size_t NetworkHelper::getInputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child) {
    bool inputIndexWasFound = false;
    size_t inputIndex;
    for (size_t i = 0; i < child->get_input_size(); ++i) {
        if (parent.get() == child->get_input_node_ptr(i)) {
            inputIndex = i;
            inputIndexWasFound = true;
            break;
        }
    }
    if (!inputIndexWasFound) {
        THROW_IE_LPT_EXCEPTION(*child) << " input index for " << parent->get_friendly_name() << " was not found";
    }

    return inputIndex;
}

std::vector<Output<Node>> NetworkHelper::getInputs(const std::shared_ptr<ngraph::Node>& node) {
    std::vector<Output<Node>> inputs(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        inputs[i] = node->get_input_node_shared_ptr(i);
    }
    return inputs;
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

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
