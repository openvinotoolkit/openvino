// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/network_helper.hpp>
#include <ngraph_ops/multiply_add.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include <ngraph/rt_info.hpp>

// #include <ie_common.h>
// #include <precision_utils.h>
// #include "cnn_network_impl.hpp"
// #include "ie_util_internal.hpp"
// #include "ie_parallel.hpp"
#include <transformations/low_precision/common/ie_lpt_exception.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {


// Return true if `type` can be castable to at least one of `type`
bool is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types) {
    for (auto another : types) {
        if (type.is_castable(another)) {
            return true;
        }
    }
    return false;
}


// Collect and return a vector with all nodes that consumes any of the `node` output
std::vector<Input<Node>> consumer_inputs(std::shared_ptr<Node> node) {
    std::vector<Input<Node>> result;
    for (const auto& output_port : node->outputs()) {
        for (const auto &input : output_port.get_target_inputs()) {
            result.push_back(input);
        }
    }
    return result;
}

std::vector<std::shared_ptr<Node>> consumers(std::shared_ptr<Node> node) {
    auto inputs = consumer_inputs(node);
    std::vector<std::shared_ptr<Node>> result(inputs.size());
    std::transform(inputs.begin(), inputs.end(), result.begin(), [](Input<Node> input){ return input.get_node()->shared_from_this(); });
    return result;
}


void NetworkHelper::updateBlobs(std::shared_ptr<opset1::FakeQuantize> quantizeLayer, int constLayerIndex, float value) {
    auto constant = std::dynamic_pointer_cast<opset1::Constant>(quantizeLayer->get_input_node_shared_ptr(constLayerIndex));
    if (!constant) {
        THROW_TRANSFORMATION_EXCEPTION << "Expected constant at " << constLayerIndex << " input for FakeQuantize node" << *quantizeLayer;
    }

    auto new_constant = std::make_shared<opset1::Constant>(constant->get_output_element_type(0), constant->get_output_shape(0), value);
    copy_runtime_info(constant, new_constant);
    replace_node(constant, new_constant);
}

int NetworkHelper::onWeightsInDepth(std::shared_ptr<Node> layer) {
    const std::vector<std::shared_ptr<Node>> children = consumers(layer);
    for (std::shared_ptr<Node> child : children) {
        // TODO: check for is_castable & get_type_info_static
        // if ((child->get_type_info().is_castable(opset1::Convolution::get_type_info_static()) ||
        //    child->get_type_info().is_castable(opset1::GroupConvolution::get_type_info_static()) ||
        //    child->get_type_info().is_castable(opset1::MatMul::get_type_info_static())) &&
        //    child->inputs().size() >= 2lu) {}

        if ((is_type<opset1::Convolution>(child) || is_type<opset1::GroupConvolution>(child) || is_type<opset1::MatMul>(child)) &&
            (child->inputs().size() >= 2lu)) {
            const std::vector<std::shared_ptr<Node>> parents = getParentsRecursivelyExceptTypes(child, {}, 1);
            for (auto parent : parents) {
                // ???
                if (parent->get_friendly_name() == layer->get_friendly_name()) {
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

std::vector<std::shared_ptr<ngraph::opset1::Constant>> NetworkHelper::transformFakeQuantizeToConst(TransformationContext& context,
                                                                        std::shared_ptr<Node> fakeQuantize,
                                                                        std::shared_ptr<opset1::Constant> weights,
                                                                        const std::string& constLayerName) {
    // TODO: update context by deleting removed layer and adding new layer if really needed
    // TODO: set proper name for a constant
    std::vector<std::shared_ptr<ngraph::opset1::Constant>> result{weights};
    copy_runtime_info(fakeQuantize, weights);
    replace_node(fakeQuantize, weights);
    return result;
#if 0 // TODO: LPT-TO-NGRAPH
    std::vector<CNNLayerPtr> constLayersToRemove;
    constLayersToRemove.reserve(fakeQuantize->insData.size());

    for (const DataWeakPtr& insDataWeak : fakeQuantize->insData) {
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input data for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input layer for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        if (!CaselessEq<std::string>()(parent->type, "Const") || (parent->insData.size() != 0lu)) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected FakeQuantize input layer type " << parent->type << " for layer '"
                               << fakeQuantize->name << "' is nullable";
        }

        constLayersToRemove.push_back(parent);
    }

    for (const CNNLayerPtr& parent : constLayersToRemove) {
        NetworkHelper::removeLayer(context.network, parent);
        context.removeLayer(*parent);
    }

    if (fakeQuantize->outData.size() != 1lu) {
        THROW_TRANSFORMATION_EXCEPTION << "FakeQuantize " << fakeQuantize->name << " has several outputs";
    }

    const DataPtr outData = fakeQuantize->outData[0];
    if (outData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "FakeQuantize output data is nullable";
    }

    // const Precision precision = outData->getPrecision();
    const auto inputTo = outData->getInputTo();
    std::vector<CNNLayerPtr> constLayers;
    for (auto it : inputTo) {
        const CNNLayerPtr child = it.second;
        if (child == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "child layer for FakeQuantize " << fakeQuantize->name << " is nullable";
        }

        constLayers.push_back(
            NetworkHelper::addConstBetween(context.network, fakeQuantize, child, weights, constLayerName));
    }

    NetworkHelper::removeLayer(context.network, fakeQuantize);
    context.removeLayer(*fakeQuantize);

    return constLayers;

#endif
}

bool NetworkHelper::IsChild(
        const std::vector<std::shared_ptr<Node>>& children,
        const std::vector<NodeTypeInfo>& layerTypes,
        const std::vector<NodeTypeInfo>& ignoreLayerTypes) {
    for (auto child : children) {
        for (auto layer_type : layerTypes) {
            if (child->get_type_info().is_castable(layer_type)) {
                return true;
            }
        }
        for (auto ignore_type : ignoreLayerTypes) {
            if (child->get_type_info().is_castable(ignore_type)) {
                if (child->outputs().size() != 1) {
                    return true;
                }
                if (IsChild(consumers(child), layerTypes, ignoreLayerTypes)) {
                    return true;
                }
            }
        }
    }
    return false;
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
Shape alignShapeForChannelDim(const Shape& shape, Rank rank) {
    assert(shape.size() == 1);
    assert(rank.is_static());
    Shape result = shape;
    result.resize(rank.get_length() - 1, 1);
    return result;
}

std::shared_ptr<Node> NetworkHelper::addScaleShiftBeforeInput(TransformationContext& context,
                                                   const Input<Node>& input,
                                                   const DequantizationDetails& dequantizationDetails,
                                                   const std::string& name) {
    auto parent = input.get_source_output().get_node_shared_ptr();
    std::string layerName = name.empty() ? (parent->get_friendly_name() + "_ScaleShift_" + input.get_node()->get_friendly_name()) : name;

    element::Type ssPrecision = context.getOriginalLayerPrecision(parent->get_friendly_name(), input.get_source_output().get_index());
    // TODO: LPT-TO-NGRAPH, not sure that it covers all valid cases
    if (ssPrecision == element::undefined) {
        ssPrecision = input.get_element_type();
    }

    auto scaleConst = std::make_shared<opset1::Constant>(
            ssPrecision,
            alignShapeForChannelDim(Shape{dequantizationDetails.channelsCount}, input.get_partial_shape().rank()),
            dequantizationDetails.scales);
    auto shiftConst = std::make_shared<opset1::Constant>(
            ssPrecision,
            alignShapeForChannelDim(Shape{dequantizationDetails.channelsCount}, input.get_partial_shape().rank()),
            dequantizationDetails.shifts);

    auto ssLayer = std::make_shared<ngraph::op::MultiplyAdd>(input.get_source_output(), scaleConst, shiftConst);

    input.get_source_output().remove_target_input(input); // Disconnect source output from input of interest
    input.replace_source_output(ssLayer->output(0)); // Connect input of interest to just created new node ssLayer output

    NetworkHelper::setOutDataPrecision(ssLayer, ssPrecision);
    return ssLayer;
}

void NetworkHelper::addDequantizationAfter(TransformationContext& context,
                                                              const Output<Node>& output,
                                                              const DequantizationDetails& dequantizationDetails) {
    auto node = output.get_node_shared_ptr();

    // TODO: provide consumer_inputs for a single output port and replace here
    auto children = consumer_inputs(node);

    std::string nameForResult = node->get_friendly_name();
    for (auto child : children) {
        std::string nameForDequantize;
        if (child.get_node()->get_type_info().is_castable(opset1::Result::get_type_info_static())) {
            if (nameForDequantize.empty()) {
                // TODO: not a regular situation when we have more than one Result for FQ or we don't have friendly_name for FQ
            } else {
                nameForDequantize = nameForResult;
                nameForResult.clear();  // use only once
            }
        }
        auto dequantizationLayer = addScaleShiftBeforeInput(
                context,
                child,
                dequantizationDetails,
                nameForDequantize);
        context.dequantizationLayersNames.insert(dequantizationLayer->get_friendly_name());
    }
}

std::vector<std::shared_ptr<Node>> NetworkHelper::getChildrenRecursivelyExceptTypes(
        std::shared_ptr<Node> layer, const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes) {
    std::vector<std::shared_ptr<Node>> children;
    for (auto child : consumers(layer)) {
        if (is_castable_to_one_of(child->get_type_info(), exceptionLayerTypes)) {
            const std::vector<std::shared_ptr<Node>> tmpChildren = getChildrenRecursivelyExceptTypes(child, exceptionLayerTypes);
            children.insert(children.end(), tmpChildren.begin(), tmpChildren.end());
        }
        children.push_back(child);
    }
    return children;
}

void NetworkHelper::removeLayer(std::shared_ptr<Node> layer) {
    ngraph::replace_output_update_name(layer->output(0), layer->input_value(0));
}

std::shared_ptr<ngraph::opset1::Constant> NetworkHelper::quantizeWeights(
        std::shared_ptr<Node> quantize,
        const bool roundValues,
        const ngraph::element::Type precision) {
    std::cerr << "[ ERROR ] " << __FILE__ << ":" << __LINE__ << '\n';
    // FIXME: this is just a placeholder
    return std::make_shared<ngraph::opset1::Constant>(quantize->get_input_element_type(0), quantize->get_input_shape(0), 5);
#if 0 // TODO: LPT-TO-NGRAPH
    if (quantize.insData.size() != 5lu) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected inputs count: " << quantize.insData.size();
    }
    for (int i = 0; i < quantize.insData.size(); i++)
        if (quantize.insData[i].lock() == nullptr)
            THROW_TRANSFORMATION_EXCEPTION << "Invalid input data for layer '" << quantize.name << "' with index " << i;

    const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
    if (sourceBlob == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "weights blob is empty for " << quantize.type << " layer " << quantize.name;
    }

    const auto& sourceBlobTD = sourceBlob->getTensorDesc();
    const Precision blobPrecision = sourceBlobTD.getPrecision();

    auto targetBlobPrecision = precision == Precision::UNSPECIFIED ? blobPrecision : precision;
    if (targetBlobPrecision != Precision::FP32 && targetBlobPrecision != Precision::FP16 &&
        targetBlobPrecision != Precision::I8 && targetBlobPrecision != Precision::U8)
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected precision: " << precision;

    Blob::Ptr targetBlob = make_blob_with_precision(TensorDesc(targetBlobPrecision, sourceBlobTD.getDims(), sourceBlobTD.getLayout()));
    targetBlob->allocate();

    quantizeBlob(quantize, targetBlob, roundValues);

    return targetBlob;
#endif
}

//std::shared_ptr<opset1::Add> decomposeMultiplyAdd(std::shared_ptr<op::MultiplyAdd> multiplyAdd) {
//    using namespace std;
//    using namespace ngraph::op;
//    // FIXME: need to modify data_type on output to be aligned with MultiplyAdd output
//    // it is fundamental limitation of TypeRelaxed approach when constructing new graphs
//    //NetworkHelper::setOutDataPrecision(multiplyAdd->input_value(0).get_node_shared_ptr(), multiplyAdd->get_output_element_type(0));
//    AutoReplaceInputTypes<Node> auto_type(*multiplyAdd, multiplyAdd->get_output_element_type(0));
//    auto multiply = make_shared<TypeRelaxed<opset1::Multiply>>(
//            opset1::Multiply(multiplyAdd->input_value(0), multiplyAdd->input_value(1)), multiplyAdd->get_output_element_type(0));
//    auto add = make_shared<opset1::Add>(multiply, multiplyAdd->input_value(2));
//    copy_runtime_info(multiplyAdd, {multiply, add});
//    add->set_friendly_name(multiplyAdd->get_friendly_name());
//    replace_node(multiplyAdd, add);
//    return add;
//}

std::shared_ptr<opset1::Multiply> swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply) {
    // Multiply --> Add(addAfterMultiply)  ==>  Add(new) --> Multiply(new)
    // That means x*a + b ==> (x + b/a)*a; tries to fold b/a
    auto x = addAfterMultiply->input_value(0).get_node()->input_value(0);
    auto a = addAfterMultiply->input_value(0).get_node()->input_value(1);
    auto b = addAfterMultiply->input_value(1);
    auto bDivA = std::make_shared<opset1::Divide>(b, a);
    OutputVector foldedTerm;
    if (bDivA->constant_fold(foldedTerm)) {
        assert(foldedTerm.size() == 1);
        auto addTerm = as_type_ptr<opset1::Constant>(foldedTerm[0].get_node_shared_ptr());
        // TODO: is it useful to optimize here?
#if 0
        if (isScalarLike(addTerm) && addTerm->cast_vector<float>()[0] == 0) {
            foldedTerm.clear();
        } else {
#endif
            replace_node(bDivA, foldedTerm);
#if 0
        }
#endif
    } else {
        foldedTerm = {bDivA->output(0)};
    }
    op::AutoReplaceInputTypes<Node> auto_type(*addAfterMultiply->input_value(0).get_node(), addAfterMultiply->get_output_element_type(0));
    Output<Node> newMultiplyInput;
    if (!foldedTerm.empty()) {
        auto newAdd = std::make_shared<op::TypeRelaxed<opset1::Add>>(opset1::Add(x, foldedTerm[0]),
                                                                     addAfterMultiply->get_output_element_type(0));
        newMultiplyInput = newAdd->output(0);
    } else {
        newMultiplyInput = x;
    }
    auto newMultiply = std::make_shared<opset1::Multiply>(newMultiplyInput, a);
    replace_node(addAfterMultiply, newMultiply);
    return newMultiply;
}

bool isScalarLike(std::shared_ptr<opset1::Constant> constant) {
#if 1
    return constant->get_all_data_elements_bitwise_identical();
#else
    // FIXME: work for floats only
    const auto scalesBuffer = constant->cast_vector<float>();
    size_t scalesBufferSize = shape_size(constant->get_output_shape(0));

    for (size_t i = 1ul; i < scalesBufferSize; ++i) {
        if (scalesBuffer[i - 1ul] != scalesBuffer[i]) {
            return false;
        }
    }
    return true;
#endif
}

std::shared_ptr<opset1::Constant> distillToScalar(std::shared_ptr<opset1::Constant> constant) {
    assert(isScalarLike(constant));
    return std::make_shared<opset1::Constant>(constant->get_element_type(), Shape{}, constant->get_data_ptr());
}

std::shared_ptr<Node> getConstantInput(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> constant1 = as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr());
    if (!constant1) {
        constant1 = as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr());
    }
    return constant1;
}


std::shared_ptr<ngraph::opset1::Multiply> optimizeMultipliesAfter(std::shared_ptr<Node> node) {
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

std::shared_ptr<opset1::Constant> roundWithTolerance(std::shared_ptr<Node> node, element::Type target_type, float tolerance) {
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
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(std::shared_ptr<opset1::FakeQuantize> fq,
                                            element::Type precision,
                                            float min,
                                            float max) {
    using std::make_shared;

    // Now calculate scales and shifts according to given shapes -- all operations in ngraph
    // TODO: why shape is empty?
    // auto newMin = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    // auto newMax = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    auto newMin = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newMax = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    auto outputLow = fq->input_value(3);
    auto outputHigh = fq->input_value(4);

    std::shared_ptr<Node> scale;
    std::shared_ptr<Node> shift;

    // TODO: threshold values have to used here to avoid shifts

    if (precision.is_signed()) {
        // I8
        scale = fold<opset1::Divide>(
            fold<opset1::Subtract>(outputHigh, outputLow),
            fold<opset1::Subtract>(newMax, newMin));

        // TODO: too complex - double check
        shift = fold<opset1::Divide>(
            fold<opset1::Divide>(
                fold<opset1::Subtract>(fold<opset1::Multiply>(newMax, outputLow), fold<opset1::Multiply>(newMin, outputHigh)),
                fold<opset1::Subtract>(newMin, newMax)),
            scale);
    } else {
        // U8
        scale = fold<opset1::Divide>(
            fold<opset1::Subtract>(outputHigh, outputLow),
            fold<opset1::Subtract>(newMax, newMin));

        shift = fold<opset1::Divide>(
            fold<opset1::Subtract>(std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, std::vector<float>({ 0 })), outputLow),
            scale);
    }

    // Build a substitution sub-graph:

    // TODO: question: why we need fold?
    auto newFQ = fold_fake_quantize<opset1::FakeQuantize>(
    // std::shared_ptr<opset1::FakeQuantize> newFQ = std::make_shared<ngraph::op::TypeRelaxed<opset1::FakeQuantize>>(
            fq->input_value(0),
            fq->input_value(1),
            fq->input_value(2),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast());
    // TODO: for debuging only - remove later
    newFQ->set_friendly_name(fq->get_friendly_name() + "_original");

    //NetworkHelper::setOutDataPrecision(newFQ, precision);

    // auto dequantize = make_shared<ngraph::op::MultiplyAdd>(
    //        make_shared<opset1::Convert>(
    //                fold<opset1::Convert>(newFQ, precision),
    //                fq->get_output_element_type(0)), scale, shift);

    std::shared_ptr<Node> convert = fold<opset1::Convert>(newFQ, precision);
    //std::shared_ptr<Node> convert = std::make_shared<opset1::Convert>(newFQ, fq->get_output_element_type(0));


    std::shared_ptr<opset1::Constant> shiftConst = as_type_ptr<opset1::Constant>(shift);
    if (isScalarLike(shiftConst)) {
        auto scalar = distillToScalar(shiftConst);
        if (op::util::constantIsEqualTo(scalar, 0)) {
            shift = nullptr;
        }
    }

    // convert,
    //make_shared<opset1::Convert>(fold<opset1::Convert>(newFQ, precision), fq->get_output_element_type(0)),
    std::shared_ptr<ngraph::Node> convert2 = make_shared<opset1::Convert>(convert, fq->get_output_element_type(0));

    std::shared_ptr<ngraph::Node> sub = shift == nullptr ? nullptr :
        make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
        // auto sub = make_shared<ngraph::opset1::Subtract>(
            convert2,
            // newFQ,
            shift);

    // const auto subShape = sub->get_shape();
    // const auto scaleShape = scale->get_shape();
    auto dequantize = make_shared<ngraph::opset1::Multiply>(sub == nullptr ? convert2 : sub, scale);

    auto outputs = newFQ->get_outputs();
    //NetworkHelper::setOutDataPrecision(newFQ, precision);

    // sub->set_overriden_output_type(element::f32);
    // std::dynamic_pointer_cast<ngraph::Node>(sub)->validate_and_infer_types();

    replace_node(fq, dequantize);
    // Make type-relaxed node

    return std::make_tuple(newFQ, dequantize);
}

std::shared_ptr<opset1::FakeQuantize> updateFakeQuantize(
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

FakeQuantizeDequantization createDequantization(
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

FakeQuantizeDequantization createDequantizationFromFakeQuantize(std::shared_ptr<opset1::FakeQuantize> fq, element::Type precision, float min, float max) {
    using std::make_shared;

    auto newMin = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newMax = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    auto outputLow = fq->input_value(3);
    auto outputHigh = fq->input_value(4);

    std::shared_ptr<Node> scale;
    std::shared_ptr<Node> shift;

    // TODO: threshold values have to used here to avoid shifts

    if (precision.is_signed()) {
        // I8
        scale = fold<opset1::Divide>(
            fold<opset1::Subtract>(outputHigh, outputLow),
            fold<opset1::Subtract>(newMax, newMin));

        // TODO: too complex - double check
        shift = fold<opset1::Divide>(
            fold<opset1::Divide>(
                fold<opset1::Subtract>(fold<opset1::Multiply>(newMax, outputLow), fold<opset1::Multiply>(newMin, outputHigh)),
                fold<opset1::Subtract>(newMin, newMax)),
            scale);
    } else {
        // U8
        scale = fold<opset1::Divide>(
            fold<opset1::Subtract>(outputHigh, outputLow),
            fold<opset1::Subtract>(newMax, newMin));

        shift = fold<opset1::Divide>(
            fold<opset1::Subtract>(std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, std::vector<float>({ 0 })), outputLow),
            scale);
    }

    // TODO: we create input here! we really need it here?
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, fq->get_output_shape(0));
    std::shared_ptr<ngraph::opset1::Convert> convert =  as_type_ptr<ngraph::opset1::Convert>(fold<opset1::Convert>(input, fq->get_output_element_type(0)));

    std::shared_ptr<ngraph::opset1::Subtract> sub = make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(convert, shift);
    sub->set_output_type(0, fq->get_output_element_type(0), sub->get_output_partial_shape(0));

    std::shared_ptr<ngraph::opset1::Multiply> multiply = make_shared<ngraph::opset1::Multiply>(sub, scale);

    return FakeQuantizeDequantization(fq, convert, sub, multiply);
}

FakeQuantizeDequantization getDequantization(const std::shared_ptr<Node> node, const size_t parentIndex) {
    std::shared_ptr<Node> dataNode = node->input_value(parentIndex).get_node_shared_ptr();

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = as_type_ptr<ngraph::opset1::Multiply>(dataNode);
    if (multiply != nullptr) {
        dataNode = multiply->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<opset1::Subtract>(dataNode);
    if (subtract != nullptr) {
        dataNode = subtract->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode);
    if (convert != nullptr) {
        dataNode = convert->get_input_node_shared_ptr(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, multiply);
}

std::shared_ptr<Node> optimizeSubtract(std::shared_ptr<opset1::Subtract> add) {
    auto convertOnAdd = add->input_value(0).get_node_shared_ptr();
    if (as_type_ptr<opset1::Convert>(convertOnAdd) == nullptr) {
        return add;
    }

    // TODO: replace assert to condition and omit conversion part if there is no convert
    // TODO: also check convertInputType to understand if we really want to propagate type
    assert(as_type_ptr<opset1::Convert>(convertOnAdd));
    auto convertInputType = convertOnAdd->get_input_element_type(0);

    auto data = convertOnAdd->input_value(0);
    auto shift = add->input_value(1).get_node_shared_ptr();
    auto roundedShift = roundWithTolerance(shift, convertInputType);

    std::shared_ptr<Node> replacement;
    if (roundedShift->get_element_type() == convertInputType) {
        // Propagate convertInputType down
        replacement = std::make_shared<opset1::Subtract>(data, roundedShift);
        replace_node(add, replacement);
    } else {
        // Try to represent it as data - (-b)
        roundedShift = roundWithTolerance(fold<opset1::Negative>(shift), convertInputType);
        if (roundedShift->get_element_type() == convertInputType) {
            // Assuming Subtract will go out of representable set of values for target type
            // So keep the original data type (likely not integer)
            replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
                    opset1::Subtract(data, roundedShift),
                    convertOnAdd->get_output_element_type(0));
            replace_node(add, replacement);
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

void moveDequantization(
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

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
