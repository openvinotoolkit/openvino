// Copyright (C) 2018-2020 Intel Corporation
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

#include "transformations/low_precision/common/dequantization_details.hpp"
#include "transformation_context.hpp"
#include "quantization_details.hpp"
#include "ngraph_ops/multiply_add.hpp"
#include "transformations/utils/utils.hpp"
#include "common/fake_quantize_dequantization.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {


// Return true if `type` can be castable to at least one of `type`
bool is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types);

std::vector<Input<Node>> consumer_inputs(std::shared_ptr<Node> node);

// Collect and return a vector with all nodes that consumes any of the `node` output
std::vector<std::shared_ptr<Node>> consumers(std::shared_ptr<Node> node);

Shape alignShapeForChannelDim(const Shape& shape, Rank rank);

/**
    * @brief NetworkHelper class encapsulates manipulations with CNN Network.
    */
class TRANSFORMATIONS_API NetworkHelper {
public:
#if 0 // TODO LPT-TO-NGRAPH

    static CNNLayerPtr getLayer(const ICNNNetwork& network, const std::string& layerName);

    static Blob::Ptr makeNewBlobPtr(const TensorDesc& desc);

    static void invertFakeQuantize(const CNNLayer& fakeQuantize);

    static void updateBlobs(CNNLayer& layer, const std::string& blobName, float value);

#endif

    static void updateBlobs(std::shared_ptr<opset1::FakeQuantize> quantizeLayer, int constLayerIndex, float value);

#if 0 // TODO LPT-TO-NGRAPH

    static void updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, const std::vector<float>& values);

    static void updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values);

#endif

    // return true if at least one child uses layer on weights
    static bool onWeights(std::shared_ptr<Node> layer);

#if 0 // TODO LPT-TO-NGRAPH

    static size_t getIndex(const CNNLayer& layer);

#endif

    static std::vector<std::shared_ptr<ngraph::opset1::Constant>> transformFakeQuantizeToConst(
            TransformationContext& context,
            std::shared_ptr<Node> fakeQuantize,
            std::shared_ptr<opset1::Constant> weights,
            const std::string& constLayerName);

    template <typename OperationType>
    static void setOutDataPrecision(std::shared_ptr<OperationType>, const element::Type& precision);

#if 0 // TODO LPT-TO-NGRAPH

    static void setOutDataPrecision(const std::vector<CNNLayerPtr>& layers, const Precision& precision);

    static void setOutDataPrecision(
        const CNNLayer& beginLayer,
        const size_t branchWithEndBeforeLayer,
        const CNNLayer& endBeforeLayer,
        const Precision& precision);

#endif

    static bool IsChild(
        const std::vector<std::shared_ptr<Node>>& children,
        const std::vector<NodeTypeInfo>& layerTypes,
        const std::vector<NodeTypeInfo>& ignoreLayerTypes = {});

    static size_t  getOutputChannelsCount(std::shared_ptr<const Node> layer, bool isOnWeights = false);

#if 0 // TODO LPT-TO-NGRAPH

    static std::vector<CNNLayerPtr> getLayers(const CNNLayer& parent, const CNNLayer& child);

    static Blob::Ptr getBlob(CNNLayerPtr layer, const std::string& blobName);

    static Blob::Ptr getBlob(CNNLayer* layer, const std::string& blobName);

    static std::shared_ptr<float> getFloatData(const CNNLayerPtr& layer, const std::string& blobName);

    static std::shared_ptr<float> getFloatData(const Blob::Ptr& srcBlob);

    static bool isBlobPrecisionSupported(const Precision precision);

    static void fillBlobByFP32(Blob::Ptr& dstBlob, float value);

    static void fillBlobByFP32(Blob::Ptr& dstBlob, const float* srcData);

    static void fillBlobByFP32(const CNNLayerPtr& layer, const std::string& blobName, const float* srcData);

    static std::shared_ptr<float> convertFloatData(const float* srcData, const size_t dataSize, const Precision precision);

    static CNNLayerPtr getParent(
        const CNNLayer& layer,
        const size_t index = 0,
        const std::string& ignoreLayerType = "");

    static std::vector<CNNLayerPtr> getParents(
        const CNNLayer& layer,
        const std::string& exceptionLayerName = "");

#endif

    static std::vector<std::shared_ptr<Node>> getParentsRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes = {},
        const int portIndex = -1);

    static size_t getInputChannelsCount(std::shared_ptr<Node> layer);

    static size_t getGroupsCount(std::shared_ptr<Node> layer);

#if 0 // TODO LPT-TO-NGRAPH

    static size_t getParamOutput(const CNNLayer& layer);

    static size_t getKernelSize(const CNNLayer& layer);

    static void renameLayer(ICNNNetwork& net, const std::string& currentName, const std::string& newName);

    static CNNLayerPtr addLayer(
        TransformationContext& context,
        const CNNLayerPtr parent,
        const CNNLayerPtr child,
        const CNNLayerPtr newLayer);

    static void replaceLayer(TransformationContext& context, const CNNLayerPtr source, const CNNLayerPtr target);

#endif

    static std::shared_ptr<Node> addScaleShiftBeforeInput(
        TransformationContext& context,
        const Input<Node>& input,
        const DequantizationDetails& dequantizationDetails,
        const std::string& name = "");

    static void addDequantizationAfter(
        TransformationContext& context,
        const Output<Node>& output,
        const DequantizationDetails& dequantizationDetails);

#if 0 // TODO LPT-TO-NGRAPH

        static CNNLayerPtr addConstBetween(
        ICNNNetwork& net,
        const CNNLayerPtr layer1,
        const CNNLayerPtr layer2,
        const Blob::Ptr customBlob,
        const std::string& name);

    static void addLayerToCNNNetworkAfterData(
        DataPtr parentOutData,
        CNNLayer::Ptr layer,
        const std::string& nextLayerName,
        ICNNNetwork& net);

    IE_SUPPRESS_DEPRECATED_START
    static void fillInScaleShift(ScaleShiftLayer* layer, const size_t channels, const float* scales, const float* shifts);
    IE_SUPPRESS_DEPRECATED_END

    static std::vector<CNNLayerPtr> getChildren(const CNNLayer& layer, const std::string& exceptionLayerName = "");

#endif

    static std::vector<std::shared_ptr<Node>> getChildrenRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes = {});

#if 0 // TODO LPT-TO-NGRAPH

    static void checkConstWithBlobs(const CNNLayerPtr layer);

    static void checkQuantizeOnWeights(const CNNLayerPtr layer);

    static void updateInput(details::CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData);

    static size_t disconnectLayers(
        CNNNetworkImpl* network,
        const CNNLayerPtr& parentLayer,
        const CNNLayerPtr& childLayer);

    static size_t getInputIndex(const CNNLayerPtr& childLayer, const CNNLayerPtr& parentLayer);

#endif

    // Remove node by connecting its 0th input with 0th output
    static void removeLayer(std::shared_ptr<Node> node);

#if 0 // TODO LPT-TO-NGRAPH

    static bool isWeightsSupported(const CNNLayer& layer) noexcept;

    static Blob::Ptr getWeights(const CNNLayer& layer, const bool roundQuantizedValues);

    static Blob::Ptr getBiases(const CNNLayer& layer);

#endif

    static std::shared_ptr<ngraph::opset1::Constant> quantizeWeights(
        std::shared_ptr<Node> quantize,
        const bool roundValues,
        const ngraph::element::Type precision = ngraph::element::undefined);

#if 0 // TODO LPT-TO-NGRAPH

    static int getConstParentBranchID(const CNNLayer& layer);

    static Precision getPrecisionParent(const CNNLayer& layer);

    static Precision getPrecisionParent(const CNNLayer& layer, const size_t parentIndex);

    static DataPtr getOutData(const CNNLayer& parentLayer, const CNNLayer& childLayer);

#endif

private:
    // 1  - on weights
    // 0  - weightable layer was not found
    // -1 - on activations
    static int onWeightsInDepth(std::shared_ptr<Node> layer);

#if 0 // TODO LPT-TO-NGRAPH

    static Precision getPrecisionParent(const CNNLayer& layer, const size_t parentIndex, const bool useParentIndex);

    static Blob::Ptr getQuantizeLayerBlob(const CNNLayer& quantize) {
        if (quantize.insData.size() < 1) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected parents count for " << quantize.type << " layer " << quantize.name;
        }

        const DataPtr data = quantize.insData[0].lock();
        if (data == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "parent data is absent for " << quantize.type << " layer " << quantize.name;
        }

        IE_SUPPRESS_DEPRECATED_START
        const CNNLayerPtr blobLayer = data->getCreatorLayer().lock();
        if (blobLayer == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "parent layer is absent for " << quantize.type << " layer " << quantize.name;
        }
        IE_SUPPRESS_DEPRECATED_END

        checkConstWithBlobs(blobLayer);

        return blobLayer->blobs.begin()->second;;
    }

    static void quantizeBlob(const CNNLayer& quantize, Blob::Ptr& targetBlob, bool roundValues);

#endif
};

template <typename OperationType>
void NetworkHelper::setOutDataPrecision(std::shared_ptr<OperationType> layer, const element::Type& precision) {
    // check if it already exteded operation node
    if (auto relaxed_layer = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(layer)) {
        relaxed_layer->set_overriden_output_type(precision);
        std::dynamic_pointer_cast<ngraph::Node>(layer)->validate_and_infer_types();
    } else {
        // TODO: Make such replacements in advance for all supported polymorphic layer types
        // extend a node with new semantics: overriden output data_type
        // FIXME: OperationType should be a real type of an object, otherwise it will lead to undefined behavior
        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<OperationType>>(*layer, precision);
        copy_runtime_info(layer, replacement);
        replace_node(layer, replacement);
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

// std::shared_ptr<opset1::Add> decomposeMultiplyAdd(std::shared_ptr<op::MultiplyAdd> multiplyAdd);
std::shared_ptr<opset1::Multiply> swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply);
bool isScalarLike(std::shared_ptr<opset1::Constant> constant);
std::shared_ptr<opset1::Constant> distillToScalar(std::shared_ptr<opset1::Constant> constant);

template <typename T, typename... Args>
std::shared_ptr<Node> fold(Args&&... args) {
    auto node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded;
        if (node->constant_fold(folded)) {
            return folded[0].get_node_shared_ptr();
        }
    }
    return node;
}

template <typename T, typename... Args>
std::shared_ptr<Node> fold_reshape(Args&&... args) {
    std::shared_ptr<Node> node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded;
        if (node->input_value(0).get_node_shared_ptr()->is_constant() && node->input_value(1).get_node_shared_ptr()->is_constant()) {
            return std::make_shared<opset1::Constant>(
                    node->get_input_element_type(0),
                    Shape(as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr())->template cast_vector<size_t>()),
                    as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr())->get_data_ptr());
        }
    }
    return node;
}

template <typename T, typename... Args>
std::shared_ptr<Node> fold_fake_quantize(Args&&... args) {
    std::shared_ptr<Node> node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded;
        if (node->input_value(0).get_node_shared_ptr()->is_constant() &&
            node->input_value(1).get_node_shared_ptr()->is_constant() &&
            node->input_value(2).get_node_shared_ptr()->is_constant() &&
            node->input_value(3).get_node_shared_ptr()->is_constant() &&
            node->input_value(4).get_node_shared_ptr()->is_constant() &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr()), 0) &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(2).get_node_shared_ptr()), 254) &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(3).get_node_shared_ptr()), -127) &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(4).get_node_shared_ptr()), 127)) {
            return fold<opset1::Add>(node->input_value(0), node->input_value(3));
        }

        // distillToScalar

        // if (node->input_value(0).get_node_shared_ptr()->is_constant() &&
        //    node->input_value(1).get_node_shared_ptr()->is_constant() &&
        //    node->input_value(2).get_node_shared_ptr()->is_constant() &&
        //    node->input_value(3).get_node_shared_ptr()->is_constant() &&
        //    node->input_value(4).get_node_shared_ptr()->is_constant() &&
        //    op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr()), -128) &&
        //    op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(2).get_node_shared_ptr()), 127) &&
        //    op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(3).get_node_shared_ptr()), -128) &&
        //    op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(4).get_node_shared_ptr()), 127)) {
        //    return fold<opset1::Add>(node->input_value(0), node->input_value(3));
        // }

        // return fold<opset1::Add>(node->input_value(0), node->input_value(3));
        // return node->input_value(3);
    }
    return node;
}


std::shared_ptr<Node> getConstantInput(std::shared_ptr<Node> node);

// Optimizes the series of multiplies after a given output port
std::shared_ptr<ngraph::opset1::Multiply> optimizeMultipliesAfter(std::shared_ptr<Node> multiply);

std::shared_ptr<opset1::Constant> roundWithTolerance(std::shared_ptr<Node> node, element::Type target_type, float tolerance = 1e-5);


std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq, element::Type precision, float min, float max);

std::shared_ptr<opset1::FakeQuantize> updateFakeQuantize(std::shared_ptr<opset1::FakeQuantize> fq, element::Type precision, float min, float max);

FakeQuantizeDequantization createDequantization(
    const float dequantizationScale,
    const float dequantizationShift,
    const ngraph::element::Type originalPrecision,
    const ngraph::Shape dataNodeOutputShape,
    element::Type precision,
    float min,
    float max);

FakeQuantizeDequantization createDequantizationFromFakeQuantize(std::shared_ptr<opset1::FakeQuantize> fq, element::Type precision, float min, float max);

FakeQuantizeDequantization getDequantization(std::shared_ptr<Node> node);

std::shared_ptr<Node> optimizeSubtract(std::shared_ptr<opset1::Subtract> add);

void moveDequantization(
    const std::shared_ptr<ngraph::Node> operation,
    const std::shared_ptr<ngraph::Node> dequantization,
    const std::shared_ptr<ngraph::Node> scalesConst = nullptr,
    const std::shared_ptr<ngraph::Node> shiftsConst = nullptr);

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
