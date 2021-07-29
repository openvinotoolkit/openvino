// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>

#include "rt_info/shared_value_attribute.hpp"
#include "rt_info/precisions_attribute.hpp"
#include "rt_info/per_tensor_quantization_attribute.hpp"
#include "rt_info/intervals_alignment_attribute.hpp"
#include "transformation_context.hpp"
#include "quantization_details.hpp"
#include "transformations/utils/utils.hpp"
#include "common/fake_quantize_dequantization.hpp"
#include "common/ie_lpt_exception.hpp"
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
* @brief NetworkHelper class encapsulates manipulations with nGraph function.
*/
class LP_TRANSFORMATIONS_API NetworkHelper {
public:
    // Return true if `type` can be castable to at least one of `type`
    static bool is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types);

    static std::vector<Input<Node>> consumer_inputs(std::shared_ptr<Node> node);

    // returns true if at least one child is not FQ
    static bool notAllChildrensAreFQ(const NodeVector& layer);

    // Collect and return a vector with all nodes that consumes any of the `node` output
    static std::vector<std::shared_ptr<Node>> consumers(std::shared_ptr<Node> node);

    // return true if op is on a constant path
    static bool isConstantPath(const std::shared_ptr<Node>& op);

    static Shape alignShapeForChannelDim(const Shape& shape, Rank rank);

    template <typename OperationType>
    static std::shared_ptr<Node> setOutDataPrecisionForTypeRelaxed(std::shared_ptr<OperationType> operation, const element::Type& precision);

    template <typename OperationType>
    static std::shared_ptr<Node> setOutDataPrecision(std::shared_ptr<OperationType> operation, const element::Type& precision);

    // applies constant folding of operation to constant and returns the specified output
    static std::shared_ptr<opset1::Constant> foldDequantizationConstant(
        const std::shared_ptr<opset1::Constant>& foldingConstant,
        const std::shared_ptr<Node>& operation,
        const size_t outIdx = 0);

    static size_t getOutputChannelsCount(std::shared_ptr<const Node> layer, bool isOnWeights = false);

    static std::vector<std::shared_ptr<Node>> getParentsRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes = {},
        const int portIndex = -1);

    static size_t getInputChannelsCount(std::shared_ptr<Node> layer);

    static size_t getGroupsCount(std::shared_ptr<Node> layer);

    // Remove node by connecting its 0th input with 0th output
    static void removeLayer(std::shared_ptr<Node> node);

    static std::shared_ptr<Node> swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply, const int multiplyBranch);

    static void copyInfo(const std::vector<std::shared_ptr<Node>>& sources, const std::vector<std::shared_ptr<Node>>& targets);

    static void copyInfo(const std::vector<std::shared_ptr<Node>>& sources, const std::shared_ptr<Node>& target);

    static void copyInfo(const std::shared_ptr<Node>& source, const std::shared_ptr<Node>& target);

    static void cleanRunTimeInfo(const std::shared_ptr<Node>& layer);

    static bool isScalarLike(std::shared_ptr<opset1::Constant> constant);

    static bool isZero(std::shared_ptr<opset1::Constant> constant);

    static std::shared_ptr<opset1::Constant> toScalar(std::shared_ptr<opset1::Constant> constant);

    static std::shared_ptr<Node> getConstantInput(std::shared_ptr<Node> node);

    static int getConstantInputIndex(std::shared_ptr<Node> node);

    static std::vector<size_t> updateReshapeValues(
        const Shape& elementwiseConstantShape,
        const Shape& elementwiseShape,
        const std::vector<size_t>& reshapeValues);

    // Optimizes the series of multiplies after a given output port
    static std::shared_ptr<ngraph::opset1::Multiply> optimizeMultipliesAfter(std::shared_ptr<Node> multiply);

    static std::shared_ptr<opset1::Constant> round(std::shared_ptr<Node> node, element::Type target_type);

    static std::shared_ptr<opset1::FakeQuantize> composeFakeQuantize(const std::shared_ptr<opset1::FakeQuantize>& fq);

    static std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(
        std::shared_ptr<opset1::FakeQuantize> fq,
        const element::Type precision,
        const float min,
        const float max,
        const bool hasZeroPoint,
        const bool updatePrecision,
        const element::Type deqPrecision = element::f32,
        const size_t outChannelsShapeIndex = 0);

    static std::shared_ptr<opset1::FakeQuantize> updateFakeQuantize(
        std::shared_ptr<opset1::FakeQuantize> fq,
        element::Type precision,
        float min,
        float max,
        const bool replace = true);

    static FakeQuantizeDequantization makeDequantization(
        const float dequantizationMul,
        const float dequantizationSub,
        const ngraph::element::Type originalPrecision,
        const ngraph::PartialShape dataNodeOutputShape,
        element::Type precision,
        const element::Type deqPrecision = element::f32,
        std::shared_ptr<ngraph::Node> input = nullptr);

    static FakeQuantizeDequantization createDequantizationFromFakeQuantize(
        std::shared_ptr<opset1::FakeQuantize> fq,
        element::Type precision,
        float min,
        float max,
        const bool hasZeroPoint,
        const bool updatePrecision,
        const element::Type deqPrecision = element::f32);

    static bool areQuantizeAndDequantizeSupportedForSubtract(const std::shared_ptr<const ngraph::Node>& node);

    static bool areQuantizeAndDequantizeSupportedForMultiply(const std::shared_ptr<const ngraph::Node>& node);

    static bool isQuantizeSupported(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize);

    static FakeQuantizeDequantization getDequantization(const std::shared_ptr<Node>& node, const size_t parentIndex = 0ul, const bool inPlace = false);

    static FakeQuantizeDequantization getDequantizationBelow(const std::shared_ptr<Node>& node, const bool convertIsMandatory = false);

    static FakeQuantizeDequantization normalizeDequantization(FakeQuantizeDequantization dequantization);

    static std::shared_ptr<opset1::Constant> normalizeDequantizationShape(const std::shared_ptr<Node>& eltwise);

    // 1. remove Convert if possible
    // 2. optimize Constant if possible
    // 3. remove Subtract if Constant on the second branch is zero
    static std::shared_ptr<Node> optimizeSubtract(std::shared_ptr<opset1::Subtract> add);

    class InsertDequantizationResult {
    public:
        InsertDequantizationResult(
            const std::shared_ptr<Node>& newOperation,
            const std::shared_ptr<Node>& lastDequantization) : newOperation(newOperation), lastDequantization(lastDequantization) {}

        std::shared_ptr<Node> newOperation;
        std::shared_ptr<Node> lastDequantization;
    };

    static InsertDequantizationResult moveDequantizationAfter(
        const std::shared_ptr<ngraph::Node>& operation,
        const FakeQuantizeDequantization& dequantization,
        const bool updatePrecision,
        const bool moveSubtract);

    static bool checkConstantValuePrecision(const element::Type expectedPrecision, const std::shared_ptr<Node>& constant);

    static size_t getChildInputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child);

    static size_t getParentOutputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child);

    static FakeQuantizeDequantizationValues createEmptyValues(const FakeQuantizeDequantization& dequantization);

    static bool isZeroConst(const std::shared_ptr<Node>& node);
    static bool checkZeroPoint(const std::shared_ptr<Node>& node, const DataPrecision& dataPrecision = DataPrecision());

    static std::shared_ptr<Node> toScalarIfPossible(std::shared_ptr<Node> node);

    static std::shared_ptr<Node> fold_fake_quantize(const std::shared_ptr<opset1::FakeQuantize>& fq);
    static std::shared_ptr<Node> fold_fake_quantize(const std::shared_ptr<opset1::FakeQuantize>& fq, const bool roundValues, int outChannelsShapeIndex = 0);

    static FakeQuantizeDequantization foldDequantization(const std::shared_ptr<Node>& node, const size_t branchIndex, const bool inPlace = false);

    static std::shared_ptr<ngraph::Node> separateInStandaloneBranch(std::shared_ptr<ngraph::Node> node);

    static std::shared_ptr<opset1::FakeQuantize> fuseConvert(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize);

    static std::vector<element::Type> precisionIntersection(
            const std::vector<element::Type>& v1,
            const std::vector<element::Type>& v2) noexcept;

    static bool isFQByDynamicDimension(const std::shared_ptr<opset1::FakeQuantize>& fq);

    static bool isDQByDynamicDimension(const std::shared_ptr<Node>& layer, size_t inputIdx = 0);

    static bool isPrecisionPreserved(const std::shared_ptr<ngraph::Node>& node);

    static void replaceAttributeInNodes(
        std::shared_ptr<ngraph::Function> f,
        const std::string& name,
        const std::shared_ptr<ngraph::Variant> newAttribute,
        const std::shared_ptr<ngraph::Variant> oldAttribute,
        const std::shared_ptr<ngraph::Node>& initialNode) {
        std::set<std::shared_ptr<Node>> visited;
        std::deque<std::shared_ptr<Node>> nodes;
        nodes.emplace_back(initialNode);

        while (!nodes.empty()) {
            auto node = nodes.front();
            nodes.pop_front();

            if (visited.count(node) || is_type<op::Constant>(node)) {
                continue;
            }

            visited.insert(node);

            bool handleConnectedNodes = false;
            if (NetworkHelper::isPrecisionPreserved(node) || is_type<opset1::FakeQuantize>(node)) {
                auto& rt = node->get_rt_info();

                if (node == initialNode) {
                    rt[name] = newAttribute;
                    handleConnectedNodes = true;
                } else {
                    auto it = rt.find(name);
                    if (it != rt.end()) {
                        const auto currentAttribute = it->second;
                        if (oldAttribute.get() == currentAttribute.get()) {
                            rt[name] = newAttribute;
                        }
                        handleConnectedNodes = true;
                    }
                }
            }

            if (!handleConnectedNodes) {
                continue;
            }

            if (!is_type<opset1::FakeQuantize>(node)) {
                for (size_t index = 0ul; index < node->get_input_size(); ++index) {
                    auto getInput = [](const std::shared_ptr<ngraph::Node>& node, const size_t index) {
                        const auto dequantization = NetworkHelper::getDequantization(node, index);
                        if (!dequantization.empty() &&
                            (is_type<opset1::Convert>(dequantization.data.get_node())) &&
                            is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                            const auto input = dequantization.data.get_node()->input(0);
                            return input;
                        }
                        return node->input(index);
                    };

                    const auto& input = getInput(node, index);
                    const auto& input_node = input.get_source_output().get_node_shared_ptr();

                    //const auto& input_node = input.get_source_output().get_node_shared_ptr();
                    if (visited.count(input_node) || is_type<op::Constant>(input_node)) {
                        continue;
                    }

                    nodes.push_front(input_node);
                }
            }

            for (auto& output : node->outputs()) {
                for (auto& input_value : output.get_target_inputs()) {
                    const auto& output_node = input_value.get_node()->shared_from_this();
                    if (visited.count(output_node) || is_type<op::Constant>(output_node)) {
                        continue;
                    }

                    nodes.push_front(output_node);
                }
            }
        }
    }

    template <typename SharedValueType, typename SharedAttributeType>
    static void reassign(
        const std::shared_ptr<SharedValueType>& sharedValue,
        const std::vector<std::weak_ptr<SharedAttributeType>>& attributes) {
        for (const auto attributeWeakPtr : attributes) {
            auto attribute = attributeWeakPtr.lock();
            if (attribute == nullptr) {
                continue;
            }
            attribute->sharedValue = sharedValue;
            sharedValue->attributes.push_back(attribute);
        }
    }

    static size_t calculateLevels(
        const float dataPrecisionMin,
        const float dataPrecisionMax,
        const float combinedIntervalLow,
        const float combinedIntervalHigh,
        const float minIntervalLow,
        const float minIntervalHigh,
        float& dequantizationMul,
        float& dequantizationSub,
        float& updatedOutputLowValue,
        float& updatedOutputHighValue);

private:
    static std::shared_ptr<Node> foldFakeQuantize(
            const std::shared_ptr<opset1::FakeQuantize>& fq,
            const bool roundValues,
            const bool roundValuesWasSet,
            int outChannelsShapeIndex = 0);

    // 1  - on weights
    // 0  - weightable layer was not found
    // -1 - on activations
    static int onWeightsInDepth(std::shared_ptr<Node> layer);
};

template <typename OperationType>
std::shared_ptr<Node> NetworkHelper::setOutDataPrecisionForTypeRelaxed(std::shared_ptr<OperationType> layer, const element::Type& precision) {
    // check if it already exteded operation node
    if (auto relaxed_layer = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(layer)) {
        relaxed_layer->set_overridden_output_type(precision);
        std::dynamic_pointer_cast<ngraph::Node>(layer)->validate_and_infer_types();
        return layer;
    } else {
        THROW_IE_LPT_EXCEPTION(*layer) << "TypeRelaxed type is expected";
    }
}

template <typename OperationType>
std::shared_ptr<Node> NetworkHelper::setOutDataPrecision(std::shared_ptr<OperationType> layer, const element::Type& precision) {
    // check if it already exteded operation node
    if (auto relaxed_layer = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(layer)) {
        relaxed_layer->set_overridden_output_type(precision);
        std::dynamic_pointer_cast<ngraph::Node>(layer)->validate_and_infer_types();
        return layer;
    } else {
        // Make such replacements in advance for all supported polymorphic layer types
        // extend a node with new semantics: overriden output data_type
        // OperationType should be a real type of an object, otherwise it will lead to undefined behavior
        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<OperationType>>(*layer, precision);
        copy_runtime_info(layer, replacement);
        replace_node(layer, replacement);
        return replacement;
    }
}

template <typename T>
std::shared_ptr<Node> make_op_pattern(const ngraph::NodeVector& args) {
    return std::make_shared<ngraph::pattern::op::Any>(element::undefined, PartialShape{}, [](std::shared_ptr<Node> n) {return !!as_type_ptr<T>(n); }, args);
}

template <typename T>
std::shared_ptr<Node> make_op_label() {
    return std::make_shared<ngraph::pattern::op::Label>(
            element::undefined,
            PartialShape{},
            [](std::shared_ptr<Node> n) {return !!as_type_ptr<T>(n); });
}

template <typename T, typename... Args>
std::shared_ptr<Node> fold(Args&&... args) {
    auto node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded(node->get_output_size());
        if (node->constant_fold(folded, node->input_values())) {
            return folded[0].get_node_shared_ptr();
        }
    }
    return node;
}

std::shared_ptr<Node> foldConvert(const Output<Node>& node, const element::Type targetPrecision);

template <typename T, typename... Args>
std::shared_ptr<Node> fold_reshape(Args&&... args) {
    std::shared_ptr<Node> node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        // issue #57985: remove fold_reshape & reuse nGraph implementation
        const auto values = as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr())->template cast_vector<int64_t>();
        if (std::any_of(values.begin(), values.end(), [](const int64_t value) { return (value == 0) || (value == -1); })) {
            return fold<opset1::Reshape>(std::forward<Args>(args)...);
        }

        OutputVector folded;
        if (is_type<opset1::Constant>(node->input_value(0).get_node_shared_ptr()) &&
            is_type<opset1::Constant>(node->input_value(1).get_node_shared_ptr())) {
            return std::make_shared<opset1::Constant>(
                    node->get_input_element_type(0),
                    Shape(as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr())->template cast_vector<size_t>()),
                    as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr())->get_data_ptr());
        }
    }
    return node;
}

template <typename T>
std::shared_ptr<ngraph::VariantWrapper<T>> getAttribute(const std::shared_ptr<Node>& inputNode) {
    auto& rt = inputNode->get_rt_info();
    auto it = rt.find(ngraph::VariantWrapper<T>::type_info.name);
    if (it == rt.end()) {
        return nullptr;
    }

    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<T>>(it->second);
    assert(attribute != nullptr);
    return attribute;
}

template <typename T>
std::shared_ptr<ngraph::VariantWrapper<T>> getAttribute(const Input<Node>& input) {
    auto& rt = input.get_rt_info();
    auto it = rt.find(ngraph::VariantWrapper<T>::type_info.name);
    if (it == rt.end()) {
        return nullptr;
    }

    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<T>>(it->second);
    assert(attribute != nullptr);
    return attribute;
}

template <typename T>
std::shared_ptr<ngraph::VariantWrapper<T>> getAttributeFromOutput(const Output<Node>& output) {
    auto& rt = output.get_rt_info();
    auto it = rt.find(ngraph::VariantWrapper<T>::type_info.name);
    if (it == rt.end()) {
        return nullptr;
    }

    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<T>>(it->second);
    assert(attribute != nullptr);
    return attribute;
}

bool isDisabled(const std::shared_ptr<Node>& node);

template <typename T, typename ... Args>
std::shared_ptr<T> make_shared_attribute(Args&& ... args) {
    std::shared_ptr<T> attribute = std::make_shared<T>(std::forward<Args>(args)...);
    attribute->sharedValue->attributes.push_back(attribute);
    return attribute;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
