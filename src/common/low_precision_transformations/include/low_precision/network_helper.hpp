// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "ov_ops/type_relaxed.hpp"

#include "rt_info/shared_value_attribute.hpp"
#include "rt_info/precisions_attribute.hpp"
#include "rt_info/quantization_granularity_attribute.hpp"
#include "rt_info/intervals_alignment_attribute.hpp"
#include "quantization_details.hpp"
#include "transformations/utils/utils.hpp"
#include "common/fake_quantize_dequantization.hpp"
#include "common/ie_lpt_exception.hpp"
#include "layer_transformation.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
* @brief NetworkHelper class encapsulates manipulations with ov::Model.
*/
class LP_TRANSFORMATIONS_API NetworkHelper {
public:
    // Return true if `type` can be castable to at least one of `type`
    static bool is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types);

    static std::vector<Input<Node>> consumer_inputs(std::shared_ptr<Node> node);

    // Collect and return a vector with all nodes that consumes any of the `node` output
    static std::vector<std::shared_ptr<Node>> consumers(std::shared_ptr<Node> node);

    // return true if op is on a constant path
    static bool isConstantPath(const std::shared_ptr<Node>& op);

    template <typename OperationType>
    static std::shared_ptr<Node> setOutDataPrecisionForTypeRelaxed(std::shared_ptr<OperationType> operation, const element::Type& precision);

    template <typename OperationType>
    static std::shared_ptr<Node> setOutDataPrecision(std::shared_ptr<OperationType> operation, const element::Type& precision);

    // applies constant folding of operation to constant and returns the specified output
    static std::shared_ptr<ov::opset1::Constant> foldDequantizationConstant(
        const std::shared_ptr<ov::opset1::Constant>& foldingConstant,
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

    static std::shared_ptr<Node> swapMultiplyAndAdd(std::shared_ptr<ov::opset1::Add> addAfterMultiply, const int multiplyBranch);

    static void copyInfo(const std::vector<std::shared_ptr<Node>>& sources, const std::vector<std::shared_ptr<Node>>& targets, bool overrideName = true);

    static void copyInfo(const std::vector<std::shared_ptr<Node>>& sources, const std::shared_ptr<Node>& target, bool overrideName = true);

    static void copyInfo(const std::shared_ptr<Node>& source, const std::shared_ptr<Node>& target, bool overrideName = true);

    static bool isScalarLike(std::shared_ptr<ov::opset1::Constant> constant);

    static bool isZero(std::shared_ptr<ov::opset1::Constant> constant);

    static std::shared_ptr<ov::opset1::Constant> toScalar(std::shared_ptr<ov::opset1::Constant> constant);

    static std::shared_ptr<Node> getConstantInput(const std::shared_ptr<const Node>& node, const bool convertIsExpected = false);

    static std::vector<size_t> updateReshapeValues(
        const Shape& elementwiseConstantShape,
        const Shape& elementwiseShape,
        const std::vector<size_t>& reshapeValues);

    // Optimizes the series of multiplies after a given output port
    static std::shared_ptr<ov::opset1::Multiply> optimizeMultipliesAfter(std::shared_ptr<Node> multiply);

    static std::shared_ptr<ov::opset1::Constant> round(std::shared_ptr<Node> node, element::Type target_type);

    static std::shared_ptr<ov::opset1::FakeQuantize> composeFakeQuantize(const std::shared_ptr<ov::opset1::FakeQuantize>& fq,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

    static std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(
        std::shared_ptr<ov::opset1::FakeQuantize> fq,
        const element::Type precision,
        const float min,
        const float max,
        const bool hasZeroPoint,
        const bool updatePrecision,
        const element::Type deqPrecision = element::f32,
        const size_t outChannelsShapeIndex = 0);

    static std::shared_ptr<ov::opset1::FakeQuantize> updateFakeQuantize(
        std::shared_ptr<ov::opset1::FakeQuantize> fq,
        element::Type precision,
        float min,
        float max,
        const bool replace = true);

    static FakeQuantizeDequantization makeDequantization(
        const float dequantizationMul,
        const float dequantizationSub,
        const ov::element::Type originalPrecision,
        const ov::PartialShape& dataNodeOutputShape,
        element::Type precision,
        const element::Type deqPrecision = element::f32,
        std::shared_ptr<ov::Node> input = nullptr);

    static std::shared_ptr<ov::Node> makeDequantizationSubtract(
        const ov::Output<ov::Node>& parent,
        const ov::Output<ov::Node>& subtract_constant);

    static bool areQuantizeAndDequantizeSupportedForSubtract(const std::shared_ptr<const ov::Node>& node,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

    static bool areQuantizeAndDequantizeSupportedForMultiply(const std::shared_ptr<const ov::Node>& node,
        const std::vector<ov::element::Type>& _defaultPrecisions = precision_set::get_int8_support());

    static bool isQuantizeSupported(const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize);

    static FakeQuantizeDequantization getDequantization(const std::shared_ptr<const Node>& node,
        const std::vector<ov::element::Type> _defaultPrecisions = precision_set::get_int8_support(),
        const size_t parentIndex = 0ul,
        const bool inPlace = false);

    static FakeQuantizeDequantization getDequantizationBelow(const std::shared_ptr<Node>& node, const bool convertIsMandatory = false);

    static FakeQuantizeDequantization normalizeDequantization(FakeQuantizeDequantization dequantization);

    static std::shared_ptr<ov::opset1::Constant> normalizeDequantizationShape(
            const std::shared_ptr<Node>& eltwise,
            const bool convertIsExpected = true);

    // 1. remove Convert if possible
    // 2. optimize Constant if possible
    // 3. remove Subtract if Constant on the second branch is zero
    static std::shared_ptr<Node> optimizeSubtract(std::shared_ptr<ov::opset1::Subtract> add);

    class InsertDequantizationResult {
    public:
        InsertDequantizationResult(
            const std::shared_ptr<Node>& newOperation,
            const std::shared_ptr<Node>& lastDequantization) : newOperation(newOperation), lastDequantization(lastDequantization) {}

        std::shared_ptr<Node> newOperation;
        std::shared_ptr<Node> lastDequantization;
    };

    static InsertDequantizationResult moveDequantizationAfter(
        const std::shared_ptr<ov::Node>& operation,
        const FakeQuantizeDequantization& dequantization,
        const bool updateOutputPrecision,
        const bool moveSubtract,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

    static InsertDequantizationResult moveDequantizationBefore(
        const std::shared_ptr<ov::Node>& operation,
        const FakeQuantizeDequantization& dequantization,
        const bool moveSubtract);

    static std::vector<std::vector<std::shared_ptr<ov::opset1::Constant>>> splitConstantsBeforeConcat(
        const std::shared_ptr<ov::Node> concat,
        const std::vector<std::shared_ptr<ov::opset1::Constant>> currConstants);

    static bool checkConstantValuePrecision(const element::Type expectedPrecision, const std::shared_ptr<Node>& constant);

    static size_t getChildInputIndex(const std::shared_ptr<ov::Node>& parent, const std::shared_ptr<ov::Node>& child);

    static size_t getParentOutputIndex(const std::shared_ptr<ov::Node>& parent, const std::shared_ptr<ov::Node>& child);

    static FakeQuantizeDequantizationValues createEmptyValues(const FakeQuantizeDequantization& dequantization,
                                                              const element::Type& precision = element::dynamic);

    static bool isZeroConst(const std::shared_ptr<Node>& node);
    static bool checkZeroPoint(const std::shared_ptr<Node>& node, const DataPrecision& dataPrecision = DataPrecision());

    static std::shared_ptr<Node> toScalarIfPossible(std::shared_ptr<Node> node);

    static std::shared_ptr<Node> fold_fake_quantize(const std::shared_ptr<ov::opset1::FakeQuantize>& fq);
    static std::shared_ptr<Node> fold_fake_quantize(const std::shared_ptr<ov::opset1::FakeQuantize>& fq, const bool roundValues);

    static FakeQuantizeDequantization foldDequantization(const std::shared_ptr<Node>& node,
        const size_t branchIndex,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support(),
        const bool inPlace = false);

    static std::shared_ptr<ov::Node> separateInStandaloneBranch(std::shared_ptr<ov::Node> node,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

    static std::shared_ptr<ov::opset1::FakeQuantize> fuseConvert(const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize);

    static std::vector<element::Type> precisionIntersection(
            const std::vector<element::Type>& v1,
            const std::vector<element::Type>& v2) noexcept;

    static bool isPrecisionPreserved(const std::shared_ptr<ov::Node>& node);

    static void insertDequantizationAfter(
        const std::shared_ptr<Node>& originalNode,
        const std::shared_ptr<Node>& dequantization,
        const std::shared_ptr<Node>& newNode);

    template <typename SharedAttribute>
    static void reassign(
        const std::shared_ptr<typename SharedAttribute::SharedValueAttribute::SharedValue>& sharedValue,
        const std::vector<std::weak_ptr<typename SharedAttribute::SharedValueAttribute>>& attributes) {
        for (const auto& attributeWeakPtr : attributes) {
            auto attribute = attributeWeakPtr.lock();
            if (attribute == nullptr) {
                continue;
            }
            attribute->sharedValue = sharedValue;
            sharedValue->addAttribute(attribute);
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

    static ov::Output<ov::Node> getSingleConsumerConstant(const ov::Output<ov::Node>& output);

    static bool checkConstantNotInf(const std::shared_ptr<Node> constant_node);

private:
    static std::shared_ptr<Node> foldFakeQuantize(
            const std::shared_ptr<ov::opset1::FakeQuantize>& fq,
            const bool roundValues,
            const bool roundValuesWasSet);

    // 1  - on weights
    // 0  - weightable layer was not found
    // -1 - on activations
    static int onWeightsInDepth(std::shared_ptr<Node> layer);
};

template <typename OperationType>
std::shared_ptr<Node> NetworkHelper::setOutDataPrecisionForTypeRelaxed(std::shared_ptr<OperationType> layer, const element::Type& precision) {
    // check if it already exteded operation node
    if (auto relaxed_layer = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(layer)) {
        relaxed_layer->set_overridden_output_type(precision);
        std::dynamic_pointer_cast<ov::Node>(layer)->validate_and_infer_types();
        return layer;
    } else {
        THROW_IE_LPT_EXCEPTION(*layer) << "TypeRelaxed type is expected";
    }
}

template <typename OperationType>
std::shared_ptr<Node> NetworkHelper::setOutDataPrecision(std::shared_ptr<OperationType> layer, const element::Type& precision) {
    // check if it already exteded operation node
    if (auto relaxed_layer = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(layer)) {
        relaxed_layer->set_overridden_output_type(precision);
        std::dynamic_pointer_cast<ov::Node>(layer)->validate_and_infer_types();
        return layer;
    } else {
        // Make such replacements in advance for all supported polymorphic layer types
        // extend a node with new semantics: overriden output data_type
        // OperationType should be a real type of an object, otherwise it will lead to undefined behavior
        auto replacement = std::make_shared<ov::op::TypeRelaxed<OperationType>>(*layer, precision);
        copy_runtime_info(layer, replacement);
        replace_node(layer, replacement);
        return replacement;
    }
}

template <typename T>
std::shared_ptr<Node> make_op_pattern(const ov::NodeVector& args) {
    return std::make_shared<ov::pass::pattern::op::Any>(
        element::dynamic,
        PartialShape{},
        [](std::shared_ptr<Node> n) {
            return !!ov::as_type_ptr<T>(n);
        },
        args);
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
    std::shared_ptr<Node> node = std::make_shared<T>(args...);
    if (node->get_output_size() == 1) {
        const auto data_const = ov::as_type_ptr<ov::opset1::Constant>(node->get_input_node_shared_ptr(0));
        const auto target_shape = ov::as_type_ptr<ov::opset1::Constant>(node->get_input_node_shared_ptr(1));
        if (data_const && target_shape) {
            return std::make_shared<ov::opset1::Constant>(node->get_input_element_type(0),
                                                      node->get_output_shape(0),
                                                      data_const->get_data_ptr());
        }
        return fold<ov::opset1::Reshape>(std::forward<Args>(args)...);
    }
    return node;
}

template <typename T>
ov::Any getAttribute(const std::shared_ptr<Node>& node) {
    auto& rt = node->get_rt_info();
    auto it = rt.find(T::get_type_info_static());
    if (it == rt.end()) {
        return {};
    }
    return it->second;
}

template <typename T>
ov::Any getAttribute(const Input<Node>& input) {
    auto& rt = input.get_rt_info();
    auto it = rt.find(T::get_type_info_static());
    if (it == rt.end()) {
        return {};
    }
    return it->second;
}

template <typename T>
ov::Any getAttributeFromOutput(const Output<Node>& output) {
    auto& rt = output.get_rt_info();
    auto it = rt.find(T::get_type_info_static());
    if (it == rt.end()) {
        return {};
    }
    return it->second;
}

bool isDisabled(const std::shared_ptr<Node>& node);

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
