// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <vector>
#include "inference_engine.hpp"
#include "details/caseless.hpp"
#include "ie_algorithm.hpp"
#include "gna-api.h"
#include "gna_permute.hpp"


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
    bool has16BOutput() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> layersWith16BOutputs = {"memory", "input", "split", "slice", "concat", "copy", "const"};
        return layersWith16BOutputs.find(layer->type) != layersWith16BOutputs.end() ||
                                                                        isActivation() ||
                                                            (isCrop() && !isCropAffined());
    }
    bool has32BOutput() const noexcept {
        IS_VALID();
        std::vector<std::function<bool()>> has32BOutputsProbes = {
            [this]() { return isFullyConnected(); },
            [this]() { return isAffineFilter(); },
            [this]() { return isConcatAlignFilter(); },
            [this]() { return isEltwise(); },
            [this]() { return isScaleShift(); },
            [this]() { return isConvolution(); },
            [this]() { return isPooling(); },
            [this]() { return isPower(); },
            [this]() { return isCropAffined(); }
        };

        for (auto && has32BOutputs : has32BOutputsProbes) {
            if (has32BOutputs()) {
                return true;
            }
        }
        return false;
    }
    static bool isBatchSizeConstrained(const std::string name) {
        static InferenceEngine::details::caseless_set<std::string> layersWithConstrains = {"memory", "convolution"};
        return layersWithConstrains.find(name) != layersWithConstrains.end();
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
             "softsign"};
        return activations.find(layer->type) != activations.end();
    }

    bool isWeightable() const noexcept {
        auto weigtable_ptr = as<const InferenceEngine::WeightableLayer*>();
        return weigtable_ptr != nullptr;
    }
    bool isConcatAlignFilter() const noexcept {
        return isOfType("ConcatAlignFilter");
    }
    bool isAffineFilter() const noexcept {
        return isOfType("AffineFilter");
    }
    bool isRelu() const noexcept {
        return isOfType("relu");
    }
    bool isConvolution() const noexcept {
        return isOfType("convolution");
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
            if (out->getInputTo().empty()) {
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
    bool isFullyConnected() const noexcept {
        return isOfType("FullyConnected") || isOfType("InnerProduct");
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
    bool isNonFunctional() const noexcept {
        return isOfType("reshape") || isOfType("squeeze") || isOfType("unsqueeze");
    }
    bool isPermute() const noexcept {
        return isOfType("permute");
    }
    // @brief this not only mathematically trivial, has some WA for kaldi case
    bool isTrivialPermute() {
        if (!isPermute()) return false;

        auto layerOrder = layer->GetParamAsInts("order");

        if (layerOrder == std::vector<int>({ 0, 3, 2, 1 })) {
            return true;  // supported case
        }
        IE_ASSERT(!layer->insData.empty());
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
            for (int j = permute.first + 1; j != permute.second; j++) {
                if (inputsOrderTransformed[j] != 1) {
                    return false;
                }
            }
            // apply permutation
            std::swap(inputsOrderTransformed[permute.first], inputsOrderTransformed[permute.second]);
        }
        return true;
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
    bool isCropAffined() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (layer);
        if (cropLayer != nullptr && !cropLayer->offset.empty()) {
            try {
                size_t cropOffset = cropLayer->offset.back() * cropLayer->precision.size();
                return (ALIGN64(cropOffset) != cropOffset);
            } catch (InferenceEngine::details::InferenceEngineException) {}
        }
        return false;
    }
    bool isCopy() const noexcept {
        return isOfType("copy");
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
