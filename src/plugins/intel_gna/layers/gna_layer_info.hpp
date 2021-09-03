// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <legacy/ie_layers.h>
#include "caseless.hpp"
#include "ie_algorithm.hpp"
#include "backend/gna_types.h"
#include "gna_permute.hpp"
#include "gna_lib_ver_selector.hpp"
#include "gna_copy_layer.hpp"
#include "gna_concat_layer.hpp"
#include "gna_graph_tools.hpp"


namespace GNAPluginNS {

/**
 * @brief detecting of const pointer for dynamic cast operations
 * @tparam T
 */
template <class T>
struct is_const_pointer : public std::false_type{
};

template <class T>
struct is_const_pointer<const T *> : public std::true_type{
};


/**
 * similar to type traits determined in standard library this trait provides details per layer type, with some attributes specific for GNA
 * we don't need to have compile time performance for this yet
 */
class LayerInfo {
    InferenceEngine::CNNLayer * layer;

#define IS_VALID() if (nullptr == layer) return false

 public:
    explicit LayerInfo(InferenceEngine::CNNLayer & layer)
        : LayerInfo(&layer) {
    }
    explicit LayerInfo(const InferenceEngine::CNNLayerPtr & layer)
        : LayerInfo(layer.get()) {
    }
    explicit LayerInfo(InferenceEngine::CNNLayer * layer)
        : layer(layer) {
    }
    bool hasMultipleInputs() const noexcept {
        IS_VALID();
        return layer->insData.size() > 1;
    }
    // The name of the funciton may be somehwat misleading
    // Explanation: when in low precision mode the listed layers have 8-bit outputs
    // and when in 16-bit input mode, they have 16-bit outputs
    bool has8BOr16BOutput() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> layersWith8BOr16BOutputs = {"memory", "input", "split", "slice", "concat", "copy", "const"};
        return layersWith8BOr16BOutputs.find(layer->type) != layersWith8BOr16BOutputs.end() ||
               isActivation() ||
               (isCrop() && !isCropAffined()) ||
               isPermute();
    }
    bool has32BOutput() const noexcept {
        IS_VALID();
        std::vector<std::function<bool()>> has32BOutputsProbes = {
            [this]() { return isFullyConnected(); },
            [this]() { return isAffineFilter(); },
            [this]() { return isConcatAlignFilter(); },
            [this]() { return isConvolutionFilter(); },
            [this]() { return isEltwise(); },
            [this]() { return isScaleShift(); },
            [this]() { return isConvolution(); },
            [this]() { return isPooling(); },
            [this]() { return isPower(); },
            [this]() { return isCropAffined(); },
            [this]() { return isGemm(); },
        };

        for (auto && has32BOutputs : has32BOutputsProbes) {
            if (has32BOutputs()) {
                return true;
            }
        }
        return false;
    }
    bool isBatchSizeConstrained() {
        static InferenceEngine::details::caseless_set<std::string> layersWithConstrains = {"memory", "convolution"};
        return layersWithConstrains.find(layer->name) != layersWithConstrains.end();
    }
    size_t getOutputBatchSize() const {
        if (!layer) {
            THROW_GNA_EXCEPTION << "layer is null";
        }
        if (!layer->outData[0]) {
            THROW_GNA_EXCEPTION << "output data of layer '" << layer->name << "' is null";
        }
        auto& dims = layer->outData[0]->getDims();
        auto layout = layer->outData[0]->getLayout();
        switch (dims.size()) {
        case 1:
            return 1;
        case 2:
            if (layout == InferenceEngine::Layout::NC) {
                return dims[0];
            } else if (layout == InferenceEngine::Layout::CN) {
                return dims[1];
            } else {
                THROW_GNA_EXCEPTION << "batch size is not define in layer '" << layer->name << "'";
            }
        case 4:
            return dims[0];
        default:
            THROW_GNA_EXCEPTION << "batch size is not define in layer '" << layer->name << "'";
        }
    }
    bool isActivation() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> activations =
            {"clamp",
             "sigmoid",
             "identity",
             "relu",
             "leakyrelu",
             "tanh",
             "prelu",
             "exp",
             "log",
             "sign",
             "abs",
             "neglog",
             "neghalflog",
             "softsign",
             "power",
             "fakequantize"};

        if (isPower()) {
            auto powerLayer = as<const InferenceEngine::PowerLayer*>();
            return powerLayer != nullptr && powerLayer->power != 1.0f;
        }

        return activations.find(layer->type) != activations.end();
    }

    bool isWeightable() const noexcept {
        auto weigtable_ptr = as<const InferenceEngine::WeightableLayer*>();
        return weigtable_ptr != nullptr;
    }
    bool isConcatAlignFilter() const noexcept {
        return isOfType("ConcatAlignFilter");
    }
    bool isLink() const noexcept {
        return isOfType("Link");
    }
    bool isAffineFilter() const noexcept {
        return isOfType("AffineFilter");
    }
    bool isConvolutionFilter() const noexcept {
        return isOfType("ConvolutionFilter");
    }
    bool isRelu() const noexcept {
        return isOfType("relu");
    }
    bool isConvolution() const noexcept {
        return isOfType("convolution");
    }
    bool isConvolutionFromUnalignedConcatFilter() const noexcept {
        if (!isConvolution()) {
            return false;
        }
        auto nextSkipPattern = [](InferenceEngine::CNNLayerPtr l) {
            auto li = LayerInfo(l);
            return li.isNonFunctional() || li.isIdentity();
        };
        auto next = InferenceEngine::CNNNetCheckNextLayerSkipCertain(layer, 0, 0, true, nextSkipPattern);
        auto nextInfo = LayerInfo(next.first);
        if (nextInfo.isConcatOrdering() && next.second.size() == 1 && next.second[0] == 2) {
            return true;
        }
        return false;
    }
    bool isPower() const noexcept {
        return isOfType("power");
    }
    bool has32BInput() const noexcept {
        IS_VALID();
        return isActivation() || isPooling();
    }
    bool isInput() const noexcept {
        return isOfType("input");
    }
    bool isOutput() const noexcept {
        for (auto& out : layer->outData) {
            if (getInputTo(out).empty()) {
                return true;
            }
        }
        return false;
    }
    bool isConst() const noexcept {
        return isOfType("const");
    }
    bool isScaleShift() const noexcept {
        IS_VALID();
        return nullptr != as<const InferenceEngine::ScaleShiftLayer*>();
    }
    bool isSyntheticScaleShift() const noexcept {
        IS_VALID();
        return layer->name.find("SyntheticScaleShift") != std::string::npos;
    }
    bool isEltwise() const noexcept {
        IS_VALID();
        return nullptr != as<const InferenceEngine::EltwiseLayer*>();
    }
    bool isEltwiseSum() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        // dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer) is validated in isEltwise function
        // coverity[var_deref_op]
        return dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer)->_operation ==
               InferenceEngine::EltwiseLayer::Sum;
    }
    bool isEltwiseSub() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        // dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer) is validated in isEltwise function
        // coverity[var_deref_op]
        return dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer)->_operation ==
            InferenceEngine::EltwiseLayer::Sub;
    }

    bool isEltwiseMul() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        // dynamic_cast<const InferenceEngine::EltwiseLayer *>(layer) is validated in isEltwise function
        // coverity[var_deref_op]
        return dynamic_cast<const InferenceEngine::EltwiseLayer*>(layer)->_operation ==
            InferenceEngine::EltwiseLayer::Prod;
    }
    bool isAbs() const noexcept {
        return isOfType("abs");
    }
    bool isIdentity() const noexcept {
        return isOfType("identity");
    }
    bool isTanh() const noexcept {
        return isOfType("tanh");
    }
    bool isSigmoid() const noexcept {
        return isOfType("sigmoid");
    }
    bool isSoftSign() const noexcept {
        return isOfType("softsign");
    }
    bool isClamp() const noexcept {
        return isOfType("clamp");
    }
    bool isFullyConnected() const noexcept {
        return isOfType("FullyConnected") || isOfType("InnerProduct");
    }
    bool isGemm() const noexcept {
        return isOfType("Gemm");
    }
    bool isSplit() const noexcept {
        return isOfType("split");
    }
    bool isSlice() const noexcept {
        return isOfType("slice");
    }
    bool isConcat() const noexcept {
        return isOfType("concat");
    }
    // Returns true for ConcatLayer which has at least 2 inputs, a single output,
    // each input's tensor dimensions are 2D i.e., [1 X], where X =! 0
    // and concatenation axis is the last one i.e. axis == 1
    bool inSimple2DConcatOnLastAxis() const noexcept {
        if (!isOfType("concat")) {
            return false;
        }
        auto concat = dynamic_cast<const InferenceEngine::ConcatLayer*>(layer);
        if (concat == nullptr) {
            return false;
        }
        if (concat->_axis != 1) {
            return false;
        }
        const auto inputSize = concat->insData.size();
        if (inputSize < 2) {
            return false;
        }
        for (auto&& i : concat->insData) {
            auto dp = i.lock();
            auto dims = dp->getDims();
            if (dims.size() != 2 || dims[0] != 1 || dims[1] == 0) {
                return false;
            }
        }
        if (concat->outData.size() != 1) {
            return false;
        }
        return true;
    }
    bool isConcatOrdering() const noexcept {
        if (!inSimple2DConcatOnLastAxis()) {
            return false;
        }
        auto outputData = layer->outData.front();
        auto concatConsumers = getInputTo(outputData);
        if (concatConsumers.size() != 2) {
            return false;
        }
        auto numOfCropAfterSubgraph = 0;
        auto isCropBeforeSubgraph = 0;
        const auto totalSize = ov::shape_size(outputData->getDims());
        const auto firstSize = ov::shape_size(layer->insData.front().lock()->getDims());
        for (auto&& consumer : concatConsumers) {
            numOfCropAfterSubgraph += LayerInfo(consumer.second).isSimple2DCropOnLastAxisAndOffsetDim(firstSize, totalSize - firstSize);
            isCropBeforeSubgraph += LayerInfo(consumer.second).isSimple2DCropOnLastAxisAndOffsetDim(0, firstSize);
        }
        if (numOfCropAfterSubgraph != 1 || isCropBeforeSubgraph != 1) {
            return false;
        }
        return true;
    }
    bool isFakeQuantize() const noexcept {
        return isOfType("FakeQuantize");
    }
    bool isNonFunctional() const noexcept {
        return isOfType("reshape") || isOfType("squeeze") || isOfType("unsqueeze") || isTrivialPermute();
    }
    bool isReshape() const noexcept {
        return isOfType("reshape");
    }
    bool isPermute() const noexcept {
        return isOfType("permute");
    }
    // @brief this not only mathematically trivial, has some WA for kaldi case
    bool isTrivialPermute() const noexcept {
        if (!isPermute()) return false;

        auto layerOrder = layer->GetParamAsInts("order");

        if (layerOrder == std::vector<int>({ 0, 3, 2, 1 })) {
            return true;  // supported case
        }
        if (layer->insData.empty()) {
            return false;  // unsupported case
        }
        auto inputs = layer->insData.begin()->lock();
        auto inputsOrder = inputs->getTensorDesc().getDims();

        // cases when all permutations happened either between 1 and X shape where no other dims in between
        auto permuteSequence = genPermutations(layerOrder.begin(), layerOrder.end());
        auto inputsOrderTransformed = inputsOrder;
        for (auto && permute : permuteSequence) {
            // check dims of permuted
            if (inputsOrderTransformed[permute.first] == 1 &&
                inputsOrderTransformed[permute.second] == 1) {
                return true;
            }
            if (inputsOrderTransformed[permute.first] != 1 &&
                inputsOrderTransformed[permute.second] != 1) {
                return false;
            }
            // check dims in between
            for (int j = std::min(permute.first, permute.second) + 1; j < std::max(permute.first, permute.second); j++) {
                if (inputsOrderTransformed[j] != 1) {
                    return false;
                }
            }
            // apply permutation
            std::swap(inputsOrderTransformed[permute.first], inputsOrderTransformed[permute.second]);
        }
        return true;
    }
    bool isNonValuesChangable() const {
        return isNonFunctional() || isSplit() || isSlice() || isConcat();
    }
    bool isPooling() const noexcept {
        return isOfType("pooling");
    }
    bool isMaxPooling() const noexcept {
        IS_VALID();
        if (!isPooling()) return false;
        return as<const InferenceEngine::PoolingLayer*>()->_type == InferenceEngine::PoolingLayer::MAX;
    }
    bool isMemory() const noexcept {
        return isOfType("memory");
    }
    bool isCrop() const noexcept {
        return isOfType("crop");
    }
    bool isSimple2DCropOnLastAxisAndOffsetDim(const size_t refOffset, const size_t refDim, const bool checkDim = true) const noexcept {
        return isSimple2DCropOnLastAxisAndOffsetDim(
            [refOffset](size_t offset) {
                return offset == refOffset;
            },
            [refDim, checkDim](size_t dim) {
                return !checkDim || dim == refDim;
            });
    }
    bool isSimple2DCropOnLastAxisAndOffsetDim(const std::function<bool(size_t)>& isOffsetValid,
                                              const std::function<bool(size_t)>& isDimValid) const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer*>(layer);
        if (cropLayer == nullptr) {
            return false;
        }
        if (cropLayer->insData.size() == 0) {
            return false;
        }
        auto input = cropLayer->insData.front().lock();
        auto offsets = cropLayer->offset;
        auto dims = cropLayer->dim;
        if (dims.size() != 2 || offsets.size() != 2) {
            return false;
        }
        if (offsets[0] != 0 || !isOffsetValid(offsets[1]) || dims[0] != 1 || !isDimValid(dims[1])) {
            return false;
        }
        return true;
    }
    bool isCropBeforeSubgraph() const noexcept {
        IS_VALID();
        if (layer->outData.size() != 1) {
            return false;
        }
        const auto shapeSize = ov::shape_size(layer->outData.front()->getDims());
        if (!isSimple2DCropOnLastAxisAndOffsetDim(0, shapeSize)) {
            return false;
        }
        auto parent = getCreatorLayer(layer->insData.front().lock()).lock();
        if (!LayerInfo(parent).inSimple2DConcatOnLastAxis()) {
            return false;
        }
        const auto dims = parent->insData.front().lock()->getDims();
        const auto firstConcatInputSize = ov::shape_size(dims);
        if (firstConcatInputSize != shapeSize) {
            return false;
        }
        return true;
    }
    bool isCropAfterSubgraph() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer*> (layer);
        if (cropLayer == nullptr || layer->insData.size() == 0) {
            return false;
        }
        auto parent = getCreatorLayer(layer->insData.front().lock()).lock();
        if (!LayerInfo(parent).inSimple2DConcatOnLastAxis()) {
            return false;
        }
        if (layer->outData.size() != 1) {
            return false;
        }
        const auto dims = parent->insData.front().lock()->getDims();
        const auto firstConcatInputSize = ov::shape_size(dims);

        const auto parentDims = parent->outData.front()->getDims();
        const auto totalParentConcatSize = ov::shape_size(parentDims);
        const auto restConcatSize = totalParentConcatSize - firstConcatInputSize;
        if (!isSimple2DCropOnLastAxisAndOffsetDim(firstConcatInputSize, restConcatSize)) {
            return false;
        }
        return true;
    }
    bool isPad() const noexcept {
        return isOfType("pad");
    }
    bool isCropFromUnalignedConcat() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer*>(layer);
        if (cropLayer == nullptr) {
            return false;
        }

        int32_t coIn = -1;
        bool concatOrdering = false;
        bool cropBeforeSubgraph = false;

        // if crop comes from unaligned concat pass then
        // no substitution of this Crop with affine layer occurs
        auto nextSkipPattern = [](InferenceEngine::CNNLayerPtr l) {
            auto li = LayerInfo(l);
            return li.isNonFunctional() || li.isPad() || li.isFullyConnected() || li.isConvolution() || li.isCrop() || li.isIdentity();
        };
        auto next = InferenceEngine::CNNNetCheckNextLayerSkipCertain(layer, 0, 0, true, nextSkipPattern);
        auto nextInfo = LayerInfo(next.first);
        if (nextInfo.isConcatOrdering() && next.second.size() == 1) {
            concatOrdering = true;
            coIn = next.second[0];
        }
        try {
            auto prevSkipPattern = [](InferenceEngine::CNNLayerPtr l) {
                auto li = LayerInfo(l);
                return li.isNonFunctional() || li.isPad() || li.isFullyConnected() || li.isConvolution();
            };
            auto prev = InferenceEngine::CNNNetPrevLayerSkipCertain(layer, 0, prevSkipPattern);
            auto prevInfo = LayerInfo(prev);
            if (prevInfo.isCropBeforeSubgraph()) {
                cropBeforeSubgraph = true;
            }
        } catch (InferenceEngine::Exception& e) {
        }
        const auto simpleCropWithNonZeroOffses = isSimple2DCropOnLastAxisAndOffsetDim(
            [](size_t offset) {
                return offset != 0;
            },
            [](size_t) {
                return true;
            });
        const auto detected = simpleCropWithNonZeroOffses && ((concatOrdering && coIn > 0) || cropBeforeSubgraph);
        return detected;
    }
    bool isCropAffined() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (layer);
        if (cropLayer != nullptr && !cropLayer->offset.empty()) {
            // if crop comes from unaligned concat pass then
            // no substitution of this Crop with affine layer occurs
            const auto cropFromUnalignedConcat = isCropFromUnalignedConcat();
            // currently crop layer only supports 2 bytes in int16 and int8 mode.
            // In fp32 mode this is not necessary but is useful for testing
            auto bytesPerCropElement = 2;
            size_t cropOffset = cropLayer->offset.back() * bytesPerCropElement;
            return !isCropAfterSubgraph() && !isCropBeforeSubgraph() && !cropFromUnalignedConcat && (ALIGN64(cropOffset) != cropOffset);
        }
        return false;
    }
    bool isCopy() const noexcept {
        return isOfType(CopyLayerName) || isOfType(DelayedCopyLayerName);
    }
    bool isCopyDelayed() const noexcept {
        return isOfType(DelayedCopyLayerName);
    }
    bool isWeightableIdentity() const noexcept {
        return isConcatAlignFilter() || isSyntheticScaleShift() || isCropAffined();
    }
    bool isFusableWithConv() const noexcept {
        return isActivation() || isMaxPooling();
    }

    bool isSynthetic() const noexcept {
        return isConcatAlignFilter() || isSyntheticScaleShift() || isConvolutionFilter() || isAffineFilter();
    }

    size_t paddingSize() const {
        static InferenceEngine::details::caseless_set<std::string> layersWithPossiblePadding = {"FullyConnected",
                                                                        "InnerProduct",
                                                                             "Pooling",
                                                                         "Convolution"};
        if (layersWithPossiblePadding.find(layer->type) != layersWithPossiblePadding.end()) {
            size_t size_without_padding = 0;
            IE_ASSERT(!layer->insData.empty());
            auto inputs = layer->insData.begin()->lock();
            if (inputs) {
                size_without_padding = InferenceEngine::details::product(begin(inputs->getTensorDesc().getDims()),
                                                                         end(inputs->getTensorDesc().getDims()));
            }
            return ALIGN(size_without_padding, 8) - size_without_padding;
        }
        return 0;
    }
    template <class T>
    typename std::enable_if<!is_const_pointer<T>::value, T>::type as() noexcept {
        return dynamic_cast<T>(layer);
    }
    template <class T>
    typename std::enable_if<is_const_pointer<T>::value, T>::type as() const noexcept {
        return dynamic_cast<T>(layer);
    }
    operator InferenceEngine::CNNLayer *() noexcept {
        return layer;
    }
    operator const InferenceEngine::CNNLayer *() const noexcept {
        return layer;
    }
    operator InferenceEngine::CNNLayerPtr () const noexcept {
        return std::shared_ptr<InferenceEngine::CNNLayer>(layer, [] (InferenceEngine::CNNLayer * p) {});
    }

 protected:
    bool isOfType(const std::string & type) const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, type);
    }
    #undef IS_VALID
};

inline std::ostream & operator <<(std::ostream &os, const LayerInfo & info) {
    os << static_cast<const InferenceEngine::CNNLayer*>(info)->name;
    return os;
}

}  // namespace GNAPluginNS
