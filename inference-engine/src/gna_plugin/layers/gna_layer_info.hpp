// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    bool isFakeQuantize() const noexcept {
        return isOfType("FakeQuantize");
    }
    bool isNonFunctional() const noexcept {
        return isOfType("reshape") || isOfType("squeeze") || isOfType("unsqueeze") || isTrivialPermute();
    }
    bool isPermute() const noexcept {
        return isOfType("permute");
    }
    // @brief this not only mathematically trivial, has some WA for kaldi case
    bool isTrivialPermute() const {
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
    bool isCropAffined() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (layer);
        if (cropLayer != nullptr && !cropLayer->offset.empty()) {
            // currently crop layer only supports 2 bytes in int16 and int8 mode.
            // In fp32 mode this is not necessary but is useful for testing
            auto bytesPerCropElement = 2;
            size_t cropOffset = cropLayer->offset.back() * bytesPerCropElement;
            return (ALIGN64(cropOffset) != cropOffset);
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
